import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_static_adjacency(num_nodes=62):
    # Approximate ring-style local topology + self-loop, row-normalized.
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    for i in range(num_nodes):
        adj[i, i] = 1.0
        for k in (1, 2, 3):
            adj[i, (i + k) % num_nodes] = 1.0
            adj[i, (i - k) % num_nodes] = 1.0
    adj = adj / adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return adj


class CSGFormerBlock(nn.Module):
    def __init__(
        self,
        input_dim=310,
        num_channels=62,
        num_bands=5,
        channel_embed_dim=8,
        node_dim=16,
        d_model=128,
        transformer_layers=1,
        transformer_heads=4,
        lambda_static=0.7,
        gamma_init=0.1,
    ):
        super(CSGFormerBlock, self).__init__()
        self.input_dim = int(input_dim)
        self.num_channels = int(num_channels)
        self.num_bands = int(num_bands)
        self.channel_embed_dim = int(channel_embed_dim)
        self.node_dim = int(node_dim)
        self.d_model = int(d_model)
        self.transformer_heads = int(transformer_heads)
        self.lambda_static = float(lambda_static)

        self.channel_embedding = nn.Embedding(self.num_channels, self.channel_embed_dim)
        self.band_mlp = nn.Sequential(
            nn.Linear(self.num_bands + self.channel_embed_dim, 16),
            nn.GELU(),
            nn.Linear(16, self.num_bands),
        )
        self.node_lift = nn.Linear(self.num_bands, self.node_dim)
        self.q_proj = nn.Linear(self.node_dim, self.node_dim)
        self.k_proj = nn.Linear(self.node_dim, self.node_dim)
        self.msg_proj = nn.Linear(self.node_dim, self.node_dim)
        self.graph_norm = nn.LayerNorm(self.node_dim)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init), dtype=torch.float32))
        self.node_score = nn.Linear(self.node_dim, 1)
        self.token_proj = nn.Linear(self.node_dim, self.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, self.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.transformer_heads,
            dim_feedforward=self.d_model * 2,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(transformer_layers))
        self.out_proj = nn.Linear(self.d_model, 64)

        self.register_buffer("a_static", build_static_adjacency(self.num_channels), persistent=True)

        self.last_a_dyn_stats = None
        self.last_pool_stats = None

    def get_gamma_value(self):
        return float(self.gamma.detach().cpu().item())

    def get_static_adj_stats(self):
        a = self.a_static
        diag = torch.diagonal(a, dim1=-2, dim2=-1)
        off_mask = ~torch.eye(self.num_channels, dtype=torch.bool, device=a.device)
        off = a[off_mask]
        return {
            "diag_mean": float(diag.mean().detach().cpu().item()),
            "offdiag_mean": float(off.mean().detach().cpu().item()),
        }

    def get_last_dyn_stats(self):
        return self.last_a_dyn_stats

    def get_last_pool_stats(self):
        return self.last_pool_stats

    def _csfa(self, x):
        bsz, t, _ = x.shape
        x_btcn = x.view(bsz, t, self.num_channels, self.num_bands)
        channel_ids = torch.arange(self.num_channels, device=x.device)
        channel_emb = self.channel_embedding(channel_ids).view(1, 1, self.num_channels, self.channel_embed_dim)
        channel_emb = channel_emb.expand(bsz, t, -1, -1)

        gate_in = torch.cat([x_btcn, channel_emb], dim=-1)
        gate_logits = self.band_mlp(gate_in)
        gate = F.softmax(gate_logits, dim=-1)
        return gate * x_btcn

    def forward(self, x):
        # x: [B, T, 310]
        bsz, t, d = x.shape
        if d != self.input_dim:
            raise ValueError("CSGFormerBlock expects input dim {}, got {}".format(self.input_dim, d))

        x_csfa = self._csfa(x)  # [B,T,62,5]
        node_feat = self.node_lift(x_csfa)  # [B,T,62,16]

        q = self.q_proj(node_feat)
        k = self.k_proj(node_feat)
        a_dyn_logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.node_dim)  # [B,T,62,62]
        a_dyn = F.softmax(a_dyn_logits, dim=-1)

        a_static = self.a_static.view(1, 1, self.num_channels, self.num_channels)
        a = self.lambda_static * a_static + (1.0 - self.lambda_static) * a_dyn

        h_msg = torch.matmul(a, node_feat)
        h_out = self.graph_norm(node_feat + self.gamma * self.msg_proj(h_msg))

        score = self.node_score(h_out).squeeze(-1)  # [B,T,62]
        alpha = F.softmax(score, dim=-1)
        token = torch.sum(alpha.unsqueeze(-1) * h_out, dim=-2)  # [B,T,16]

        token = self.token_proj(token)  # [B,T,128]
        if t > self.pos_embed.size(1):
            raise ValueError("sequence length {} exceeds positional embedding limit {}".format(t, self.pos_embed.size(1)))
        token = token + self.pos_embed[:, :t, :]
        token = self.temporal_transformer(token)
        out = self.out_proj(token)  # [B,T,64]

        with torch.no_grad():
            a_dyn_min = float(a_dyn.min().detach().cpu().item())
            a_dyn_max = float(a_dyn.max().detach().cpu().item())
            a_dyn_has_nan = bool((~torch.isfinite(a_dyn)).any().detach().cpu().item())
            alpha_max = alpha.max(dim=-1).values
            alpha_entropy = -(alpha * torch.log(alpha.clamp_min(1e-9))).sum(dim=-1)
            self.last_a_dyn_stats = {
                "min": a_dyn_min,
                "max": a_dyn_max,
                "has_nan_or_inf": a_dyn_has_nan,
            }
            self.last_pool_stats = {
                "alpha_max_mean": float(alpha_max.mean().detach().cpu().item()),
                "alpha_entropy_mean": float(alpha_entropy.mean().detach().cpu().item()),
            }
        return out
