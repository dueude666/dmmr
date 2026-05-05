import torch
import torch.nn as nn


class FeatureDistributionCalibrator(nn.Module):
    """Fixed source-feature standardization with a bounded residual gate."""

    def __init__(self, feature_dim=64, alpha=0.5, learnable_alpha=False, eps=1e-5, use_std=False):
        super(FeatureDistributionCalibrator, self).__init__()
        self.feature_dim = int(feature_dim)
        self.eps = float(eps)
        self.use_std = bool(use_std)
        self.learnable_alpha = bool(learnable_alpha)
        self.register_buffer("source_mean", torch.zeros(self.feature_dim), persistent=True)
        self.register_buffer("source_std", torch.ones(self.feature_dim), persistent=True)
        init_alpha = min(max(float(alpha), 1e-6), 1.0 - 1e-6)
        raw = torch.log(torch.tensor(init_alpha / (1.0 - init_alpha)))
        if self.learnable_alpha:
            self.alpha_raw = nn.Parameter(raw)
        else:
            self.register_buffer("alpha_raw", raw, persistent=True)
        self.last_stats = {}

    def set_stats(self, mean, std):
        if mean.shape[0] != self.feature_dim or std.shape[0] != self.feature_dim:
            raise ValueError("feature stats shape mismatch: mean {}, std {}".format(tuple(mean.shape), tuple(std.shape)))
        with torch.no_grad():
            self.source_mean.copy_(mean.to(self.source_mean.device, dtype=self.source_mean.dtype))
            self.source_std.copy_(std.clamp_min(self.eps).to(self.source_std.device, dtype=self.source_std.dtype))

    def get_alpha(self):
        return torch.sigmoid(self.alpha_raw)

    def get_alpha_value(self):
        return float(self.get_alpha().detach().cpu().item())

    def get_last_stats(self):
        return dict(self.last_stats)

    def forward(self, feat):
        feat_centered = feat - self.source_mean.view(1, -1)
        if self.use_std:
            feat_target = feat_centered / self.source_std.view(1, -1).clamp_min(self.eps)
        else:
            feat_target = feat_centered
        alpha = self.get_alpha()
        out = feat + alpha * (feat_target - feat)
        with torch.no_grad():
            delta = out - feat
            self.last_stats = {
                "alpha": self.get_alpha_value(),
                "source_mean_norm": float(self.source_mean.norm().detach().cpu().item()),
                "source_std_mean": float(self.source_std.mean().detach().cpu().item()),
                "source_std_min": float(self.source_std.min().detach().cpu().item()),
                "use_std": self.use_std,
                "feature_norm_mean": float(feat.norm(dim=-1).mean().detach().cpu().item()),
                "calibrated_norm_mean": float(out.norm(dim=-1).mean().detach().cpu().item()),
                "feature_delta_norm_mean": float(delta.norm(dim=-1).mean().detach().cpu().item()),
                "has_nan_or_inf": bool(
                    torch.isnan(out).any().item()
                    or torch.isinf(out).any().item()
                    or torch.isnan(feat_target).any().item()
                    or torch.isinf(feat_target).any().item()
                ),
            }
        return out
