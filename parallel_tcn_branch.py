import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConvBlock(nn.Module):
    def __init__(self, hidden_dim=64, kernel_size=3, dilation=1, dropout=0.1):
        super(TemporalConvBlock, self).__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=hidden_dim,
        )
        self.pointwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return residual + x


class ParallelTCNBranch(nn.Module):
    """Parallel temporal-convolution side branch fused weakly with LSTM feature."""

    def __init__(
        self,
        input_dim=310,
        feature_dim=64,
        hidden_dim=64,
        num_layers=2,
        kernel_size=3,
        dropout=0.1,
        alpha_init=0.1,
        alpha_max=0.3,
        delta_init_std=1e-2,
    ):
        super(ParallelTCNBranch, self).__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.alpha_max = float(alpha_max)
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                TemporalConvBlock(
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=2 ** i,
                    dropout=dropout,
                )
                for i in range(num_layers)
            ]
        )
        self.pool_score = nn.Linear(hidden_dim, 1)
        self.to_feature = nn.Linear(hidden_dim, feature_dim)
        self.delta_net = nn.Sequential(
            nn.Linear(feature_dim * 4, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
        )
        if float(delta_init_std) > 0:
            nn.init.normal_(self.delta_net[-1].weight, mean=0.0, std=float(delta_init_std))
            nn.init.zeros_(self.delta_net[-1].bias)
        else:
            nn.init.zeros_(self.delta_net[-1].weight)
            nn.init.zeros_(self.delta_net[-1].bias)
        self.alpha_raw = nn.Parameter(self._bounded_raw(float(alpha_init), self.alpha_max))
        self.last_stats = {}

    @staticmethod
    def _bounded_raw(target_value, max_value):
        eps = 1e-6
        ratio = min(max(target_value / max(max_value, eps), eps), 1.0 - eps)
        return torch.log(torch.tensor(ratio / (1.0 - ratio)))

    def get_alpha(self):
        return self.alpha_max * torch.sigmoid(self.alpha_raw)

    def get_alpha_value(self):
        return float(self.get_alpha().detach().cpu().item())

    def get_last_stats(self):
        return dict(self.last_stats)

    def forward(self, x, lstm_feat):
        # x: [B,T,310]
        y = x.transpose(1, 2)
        y = F.gelu(self.input_proj(y))
        for block in self.blocks:
            y = block(y)
        tokens = y.transpose(1, 2)
        scores = self.pool_score(tokens).squeeze(-1)
        attn = F.softmax(scores, dim=-1)
        pooled = torch.sum(tokens * attn.unsqueeze(-1), dim=1)
        tcn_feat = self.to_feature(pooled)
        delta_input = torch.cat(
            [lstm_feat, tcn_feat, lstm_feat - tcn_feat, torch.abs(lstm_feat - tcn_feat)],
            dim=-1,
        )
        delta = self.delta_net(delta_input)
        alpha = self.get_alpha()
        out = lstm_feat + alpha * delta

        with torch.no_grad():
            safe_attn = attn.clamp_min(1e-12)
            entropy = -(safe_attn * safe_attn.log()).sum(dim=-1)
            self.last_stats = {
                "alpha": float(alpha.detach().cpu().item()),
                "tcn_feat_norm_mean": float(tcn_feat.norm(dim=-1).mean().detach().cpu().item()),
                "lstm_feat_norm_mean": float(lstm_feat.norm(dim=-1).mean().detach().cpu().item()),
                "delta_norm_mean": float(delta.norm(dim=-1).mean().detach().cpu().item()),
                "feature_delta_norm_mean": float((out - lstm_feat).norm(dim=-1).mean().detach().cpu().item()),
                "attn_entropy_mean": float(entropy.mean().detach().cpu().item()),
                "attn_entropy_norm_mean": float((entropy / math.log(max(attn.shape[-1], 2))).mean().detach().cpu().item()),
                "attn_max_mean": float(attn.max(dim=-1).values.mean().detach().cpu().item()),
                "has_nan_or_inf": bool(
                    torch.isnan(out).any().item()
                    or torch.isinf(out).any().item()
                    or torch.isnan(attn).any().item()
                    or torch.isinf(attn).any().item()
                ),
            }
        return out
