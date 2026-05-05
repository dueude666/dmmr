import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiSourceSubjectRouter(nn.Module):
    """Route target features through a learnable source-subject memory bank."""

    def __init__(
        self,
        feature_dim=64,
        num_sources=14,
        hidden_dim=128,
        tau=1.0,
        alpha_init=-2.2,
        dropout=0.1,
        memory_init_std=0.02,
        delta_init_std=1e-3,
    ):
        super(MultiSourceSubjectRouter, self).__init__()
        self.feature_dim = int(feature_dim)
        self.num_sources = int(num_sources)
        self.tau = float(tau)

        self.subject_memory = nn.Parameter(
            torch.randn(self.num_sources, self.feature_dim) * float(memory_init_std)
        )
        self.router = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_sources),
        )
        self.delta_net = nn.Sequential(
            nn.Linear(self.feature_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.feature_dim),
        )
        self.delta_init_std = float(delta_init_std)
        if self.delta_init_std > 0:
            nn.init.normal_(self.delta_net[-1].weight, mean=0.0, std=self.delta_init_std)
            nn.init.zeros_(self.delta_net[-1].bias)
        else:
            nn.init.zeros_(self.delta_net[-1].weight)
            nn.init.zeros_(self.delta_net[-1].bias)

        self.raw_alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.last_stats = {}

    def get_alpha(self):
        return 0.1 * torch.sigmoid(self.raw_alpha)

    def get_alpha_value(self):
        return float(self.get_alpha().detach().cpu().item())

    def get_last_stats(self):
        return dict(self.last_stats)

    def set_subject_memory(self, prototypes):
        if prototypes.shape != self.subject_memory.shape:
            raise ValueError(
                "prototype shape {} does not match subject memory shape {}".format(
                    tuple(prototypes.shape), tuple(self.subject_memory.shape)
                )
            )
        with torch.no_grad():
            self.subject_memory.copy_(prototypes.to(self.subject_memory.device, dtype=self.subject_memory.dtype))

    def forward(self, feat):
        tau = max(self.tau, 1e-6)
        logits = self.router(feat)
        weights = F.softmax(logits / tau, dim=-1)
        context = torch.matmul(weights, self.subject_memory)
        delta_input = torch.cat([feat, context, feat - context, torch.abs(feat - context)], dim=-1)
        delta = self.delta_net(delta_input)
        alpha = self.get_alpha()
        feat_delta = alpha * delta
        out = feat + feat_delta

        with torch.no_grad():
            safe_weights = weights.clamp_min(1e-12)
            entropy = -(safe_weights * safe_weights.log()).sum(dim=-1)
            top1 = weights.argmax(dim=-1)
            top1_counts = torch.bincount(top1, minlength=self.num_sources)
            self.last_stats = {
                "alpha": float(alpha.detach().cpu().item()),
                "router_entropy_mean": float(entropy.mean().detach().cpu().item()),
                "router_entropy_norm_mean": float((entropy / math.log(max(self.num_sources, 2))).mean().detach().cpu().item()),
                "router_max_mean": float(weights.max(dim=-1).values.mean().detach().cpu().item()),
                "context_norm_mean": float(context.norm(dim=-1).mean().detach().cpu().item()),
                "delta_norm_mean": float(delta.norm(dim=-1).mean().detach().cpu().item()),
                "feature_delta_norm_mean": float(feat_delta.norm(dim=-1).mean().detach().cpu().item()),
                "memory_norm_mean": float(self.subject_memory.norm(dim=-1).mean().detach().cpu().item()),
                "delta_init_std": self.delta_init_std,
                "top1_counts": [int(v) for v in top1_counts.detach().cpu().tolist()],
                "has_nan_or_inf": bool(
                    torch.isnan(out).any().item()
                    or torch.isinf(out).any().item()
                    or torch.isnan(weights).any().item()
                    or torch.isinf(weights).any().item()
                ),
            }
        return out
