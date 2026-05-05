import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _bounded_raw(target_value, max_value):
    eps = 1e-6
    ratio = min(max(float(target_value) / max(float(max_value), eps), eps), 1.0 - eps)
    return torch.log(torch.tensor(ratio / (1.0 - ratio)))


class EEGConformerFusionBranch(nn.Module):
    """
    EEG-Conformer-style side branch for DMMR.

    The original ABP+LSTM feature remains the main path. This branch builds
    channel-aware spectral tokens, applies local temporal convolution plus a
    lightweight Transformer encoder, then injects a residual delta into the
    original [B,64] feature.
    """

    def __init__(
        self,
        input_dim=310,
        feature_dim=64,
        num_channels=62,
        num_bands=5,
        node_dim=32,
        d_model=128,
        num_heads=4,
        num_layers=1,
        dropout=0.1,
        alpha_init=0.25,
        alpha_max=0.8,
        delta_init_std=0.02,
        use_cls_pool=False,
        max_time_steps=64,
        use_gate_warmup=False,
        warmup_epochs=2,
        ramp_epochs=2,
    ):
        super(EEGConformerFusionBranch, self).__init__()
        assert int(input_dim) == int(num_channels) * int(num_bands)
        self.num_channels = int(num_channels)
        self.num_bands = int(num_bands)
        self.feature_dim = int(feature_dim)
        self.alpha_max = float(alpha_max)
        self.use_cls_pool = bool(use_cls_pool)
        self.max_time_steps = int(max_time_steps)
        self.use_gate_warmup = bool(use_gate_warmup)
        self.warmup_epochs = int(warmup_epochs)
        self.ramp_epochs = max(1, int(ramp_epochs))

        self.band_proj = nn.Linear(self.num_bands, int(node_dim))
        self.channel_embed = nn.Embedding(self.num_channels, int(node_dim))
        self.node_score = nn.Sequential(
            nn.Linear(int(node_dim), int(node_dim)),
            nn.Tanh(),
            nn.Linear(int(node_dim), 1),
        )
        self.token_proj = nn.Linear(int(node_dim), int(d_model))
        self.local_temporal = nn.Sequential(
            nn.Conv1d(int(d_model), int(d_model), kernel_size=3, padding=1, groups=int(d_model)),
            nn.Conv1d(int(d_model), int(d_model), kernel_size=1),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(num_heads),
            dim_feedforward=int(d_model) * 2,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, int(d_model)))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_time_steps + 1, int(d_model)))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        self.temporal_score = nn.Sequential(
            nn.Linear(int(d_model), int(d_model) // 2),
            nn.Tanh(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(d_model) // 2, 1),
        )
        self.to_feature = nn.Linear(int(d_model), self.feature_dim)
        self.fusion = nn.Sequential(
            nn.Linear(self.feature_dim * 4, self.feature_dim * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
        )
        nn.init.normal_(self.fusion[-1].weight, mean=0.0, std=float(delta_init_std))
        nn.init.zeros_(self.fusion[-1].bias)
        self.alpha_raw = nn.Parameter(_bounded_raw(alpha_init, alpha_max))
        self.last_stats = {}

    def get_alpha(self):
        return self.alpha_max * torch.sigmoid(self.alpha_raw)

    def get_alpha_value(self):
        return float(self.get_alpha().detach().cpu().item())

    def get_last_stats(self):
        return dict(self.last_stats)

    def get_inject_scale(self, current_epoch=None):
        if (not self.use_gate_warmup) or current_epoch is None:
            return 1.0
        current_epoch = int(current_epoch)
        if current_epoch < self.warmup_epochs:
            return 0.0
        if current_epoch < self.warmup_epochs + self.ramp_epochs:
            return float(current_epoch - self.warmup_epochs + 1) / float(self.ramp_epochs)
        return 1.0

    def forward(self, x, base_feat, current_epoch=None):
        bsz, steps, _ = x.shape
        xb = x.view(bsz, steps, self.num_channels, self.num_bands)
        node = self.band_proj(xb)
        channel_ids = torch.arange(self.num_channels, device=x.device)
        node = node + self.channel_embed(channel_ids).view(1, 1, self.num_channels, -1)

        node_logits = self.node_score(node).squeeze(-1)
        node_attn = F.softmax(node_logits, dim=-1)
        tokens = torch.sum(node * node_attn.unsqueeze(-1), dim=2)

        h = self.token_proj(tokens)
        local = self.local_temporal(h.transpose(1, 2)).transpose(1, 2)
        h = h + local
        if self.use_cls_pool:
            if steps > self.max_time_steps:
                raise ValueError("steps {} exceeds max_time_steps {}".format(steps, self.max_time_steps))
            cls = self.cls_token.expand(bsz, -1, -1)
            h_with_cls = torch.cat([cls, h], dim=1)
            h_with_cls = h_with_cls + self.pos_embed[:, : steps + 1, :]
            h_with_cls = self.transformer(h_with_cls)
            pooled = h_with_cls[:, 0, :]
            token_out = h_with_cls[:, 1:, :]
            time_logits = torch.bmm(token_out, pooled.unsqueeze(-1)).squeeze(-1) / math.sqrt(float(token_out.shape[-1]))
            time_attn = F.softmax(time_logits, dim=-1)
        else:
            h = self.transformer(h)
            time_logits = self.temporal_score(h).squeeze(-1)
            time_attn = F.softmax(time_logits, dim=-1)
            pooled = torch.sum(h * time_attn.unsqueeze(-1), dim=1)
        conformer_feat = self.to_feature(pooled)

        fusion_input = torch.cat(
            [base_feat, conformer_feat, base_feat - conformer_feat, torch.abs(base_feat - conformer_feat)],
            dim=-1,
        )
        delta = self.fusion(fusion_input)
        alpha = self.get_alpha()
        inject_scale = self.get_inject_scale(current_epoch)
        fused = base_feat + (alpha * float(inject_scale)) * delta

        with torch.no_grad():
            safe_node = node_attn.clamp_min(1e-12)
            node_entropy = -(safe_node * safe_node.log()).sum(dim=-1)
            safe_time = time_attn.clamp_min(1e-12)
            time_entropy = -(safe_time * safe_time.log()).sum(dim=-1)
            self.last_stats = {
                "alpha": float(alpha.detach().cpu().item()),
                "inject_scale": float(inject_scale),
                "base_feat_norm_mean": float(base_feat.norm(dim=-1).mean().detach().cpu().item()),
                "conformer_feat_norm_mean": float(conformer_feat.norm(dim=-1).mean().detach().cpu().item()),
                "delta_norm_mean": float(delta.norm(dim=-1).mean().detach().cpu().item()),
                "feature_delta_norm_mean": float((fused - base_feat).norm(dim=-1).mean().detach().cpu().item()),
                "node_attn_entropy_norm_mean": float((node_entropy / math.log(float(max(self.num_channels, 2)))).mean().detach().cpu().item()),
                "node_attn_max_mean": float(node_attn.max(dim=-1).values.mean().detach().cpu().item()),
                "time_attn_entropy_norm_mean": float((time_entropy / math.log(float(max(steps, 2)))).mean().detach().cpu().item()),
                "time_attn_max_mean": float(time_attn.max(dim=-1).values.mean().detach().cpu().item()),
                "pooling": "cls" if self.use_cls_pool else "attention",
                "has_nan_or_inf": bool(
                    torch.isnan(fused).any().item()
                    or torch.isinf(fused).any().item()
                    or torch.isnan(node_attn).any().item()
                    or torch.isinf(node_attn).any().item()
                    or torch.isnan(time_attn).any().item()
                    or torch.isinf(time_attn).any().item()
                ),
            }
        return fused
