import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_eeg_static_adjacency(num_nodes=62):
    # Fixed EEG topology prior (approximate local-neighbor graph), with self-loop.
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    for i in range(num_nodes):
        adj[i, i] = 1.0
        for k in (1, 2, 3):
            adj[i, (i + k) % num_nodes] = 1.0
            adj[i, (i - k) % num_nodes] = 1.0
    adj = adj / adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return adj


class EmTLiteV2Encoder(nn.Module):
    def __init__(
        self,
        input_dim=310,
        num_channels=62,
        num_bands=5,
        node_dim=16,
        channel_embed_dim=8,
        lambda_static=0.7,
        gamma_graph_init=0.1,
        d_model=128,
        transformer_heads=4,
        transformer_layers=1,
        use_temporal_attn_pool=True,
    ):
        super(EmTLiteV2Encoder, self).__init__()
        self.input_dim = int(input_dim)
        self.num_channels = int(num_channels)
        self.num_bands = int(num_bands)
        self.node_dim = int(node_dim)
        self.channel_embed_dim = int(channel_embed_dim)
        self.lambda_static = float(lambda_static)
        self.use_temporal_attn_pool = bool(use_temporal_attn_pool)
        self.d_model = int(d_model)

        # Step 2: channel-conditioned node lift
        self.base_proj = nn.Linear(self.num_bands, self.node_dim)
        self.channel_embedding = nn.Embedding(self.num_channels, self.channel_embed_dim)
        self.gamma_mlp = nn.Sequential(
            nn.Linear(self.channel_embed_dim, self.node_dim),
            nn.Tanh(),
        )
        self.beta_mlp = nn.Linear(self.channel_embed_dim, self.node_dim)

        # Step 3: static-guided dynamic graph
        self.q_proj = nn.Linear(self.node_dim, self.node_dim)
        self.k_proj = nn.Linear(self.node_dim, self.node_dim)
        self.msg_proj = nn.Linear(self.node_dim, self.node_dim)
        self.graph_norm = nn.LayerNorm(self.node_dim)
        self.gamma_graph = nn.Parameter(torch.tensor(float(gamma_graph_init), dtype=torch.float32))
        self.register_buffer("a_static", build_eeg_static_adjacency(self.num_channels), persistent=True)

        # Step 4: node attentive pooling
        self.node_score = nn.Linear(self.node_dim, 1)

        # Step 5: temporal transformer
        self.to_model = nn.Linear(self.node_dim, self.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, self.d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(transformer_heads),
            dim_feedforward=self.d_model * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(enc_layer, num_layers=int(transformer_layers))

        # Step 6: temporal attention pooling
        self.temporal_score = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.out_proj = nn.Linear(self.d_model, 64)

        self.last_stats = {}

    def get_gamma_graph_value(self):
        return float(self.gamma_graph.detach().cpu().item())

    def get_static_adj_stats(self):
        a = self.a_static
        diag = torch.diagonal(a, dim1=-2, dim2=-1)
        off_mask = ~torch.eye(self.num_channels, dtype=torch.bool, device=a.device)
        off = a[off_mask]
        return {
            "diag_mean": float(diag.mean().detach().cpu().item()),
            "offdiag_mean": float(off.mean().detach().cpu().item()),
        }

    def get_last_stats(self):
        return self.last_stats

    def forward(self, x):
        # x: [B, T, 310]
        bsz, t, dim = x.shape
        if dim != self.input_dim:
            raise ValueError("EmTLiteV2Encoder expects input_dim {}, got {}".format(self.input_dim, dim))

        x = x.view(bsz, t, self.num_channels, self.num_bands)  # [B,T,62,5]

        base_feat = self.base_proj(x)  # [B,T,62,node_dim]
        ch_idx = torch.arange(self.num_channels, device=x.device)
        ch_emb = self.channel_embedding(ch_idx)  # [62,embed]
        raw_gamma = self.gamma_mlp(ch_emb)  # [62,node_dim]
        gamma = 1.0 + 0.1 * raw_gamma
        beta = self.beta_mlp(ch_emb)  # [62,node_dim]
        gamma = gamma.view(1, 1, self.num_channels, self.node_dim)
        beta = beta.view(1, 1, self.num_channels, self.node_dim)
        node_feat = gamma * base_feat + beta  # [B,T,62,node_dim]

        q = self.q_proj(node_feat)
        k = self.k_proj(node_feat)
        a_dyn_logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.node_dim)  # [B,T,62,62]
        a_dyn = F.softmax(a_dyn_logits, dim=-1)
        a_static = self.a_static.view(1, 1, self.num_channels, self.num_channels)
        a = self.lambda_static * a_static + (1.0 - self.lambda_static) * a_dyn

        h_msg = torch.matmul(a, node_feat)
        h_out = self.graph_norm(node_feat + self.gamma_graph * self.msg_proj(h_msg))  # [B,T,62,node_dim]

        node_score = self.node_score(h_out).squeeze(-1)  # [B,T,62]
        node_alpha = F.softmax(node_score, dim=-1)
        temporal_tokens = torch.sum(node_alpha.unsqueeze(-1) * h_out, dim=-2)  # [B,T,node_dim]

        z = self.to_model(temporal_tokens)  # [B,T,d_model]
        if t > self.pos_embed.size(1):
            raise ValueError("time length {} exceeds positional embedding length {}".format(t, self.pos_embed.size(1)))
        z = z + self.pos_embed[:, :t, :]
        z = self.temporal_encoder(z)  # [B,T,d_model]

        if self.use_temporal_attn_pool:
            t_score = self.temporal_score(z).squeeze(-1)  # [B,T]
            t_alpha = F.softmax(t_score, dim=-1)
            global_feat = torch.sum(t_alpha.unsqueeze(-1) * z, dim=1)  # [B,d_model]
        else:
            t_alpha = torch.full((bsz, t), 1.0 / float(t), device=x.device, dtype=z.dtype)
            global_feat = z.mean(dim=1)

        out = self.out_proj(global_feat)  # [B,64]
        hn = out.unsqueeze(0)  # mimic LSTM interface
        cn = torch.zeros_like(hn)

        with torch.no_grad():
            self.last_stats = {
                "gamma_min": float(gamma.min().detach().cpu().item()),
                "gamma_max": float(gamma.max().detach().cpu().item()),
                "beta_min": float(beta.min().detach().cpu().item()),
                "beta_max": float(beta.max().detach().cpu().item()),
                "a_dyn_min": float(a_dyn.min().detach().cpu().item()),
                "a_dyn_max": float(a_dyn.max().detach().cpu().item()),
                "a_dyn_has_nan_or_inf": bool((~torch.isfinite(a_dyn)).any().detach().cpu().item()),
                "node_alpha_max_mean": float(node_alpha.max(dim=-1).values.mean().detach().cpu().item()),
                "node_alpha_entropy_mean": float((-(node_alpha * torch.log(node_alpha.clamp_min(1e-9))).sum(dim=-1)).mean().detach().cpu().item()),
                "temp_alpha_max_mean": float(t_alpha.max(dim=-1).values.mean().detach().cpu().item()),
                "temp_alpha_entropy_mean": float((-(t_alpha * torch.log(t_alpha.clamp_min(1e-9))).sum(dim=-1)).mean().detach().cpu().item()),
            }
        return out, hn, cn
