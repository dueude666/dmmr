import torch
import torch.nn as nn
import torch.nn.functional as F


class HemiAsymmetryFusion(nn.Module):
    def __init__(self, feature_dim=64, hidden_dim=128, dropout=0.1, gate_init=-2.2):
        super(HemiAsymmetryFusion, self).__init__()
        self.feature_dim = int(feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim * 2, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.feature_dim),
        )
        nn.init.normal_(self.mlp[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.mlp[-1].bias)
        self.raw_alpha = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))
        self.last_stats = {}

    def get_alpha_value(self):
        alpha = 0.1 * torch.sigmoid(self.raw_alpha)
        return float(alpha.detach().cpu().item())

    def get_last_stats(self):
        return self.last_stats

    def forward(self, f_full, f_left, f_right):
        diff_signed = f_left - f_right
        diff_abs = torch.abs(diff_signed)
        hemi_in = torch.cat([diff_signed, diff_abs], dim=-1)
        hemi_side = self.mlp(hemi_in)
        alpha = 0.1 * torch.sigmoid(self.raw_alpha)
        f_fused = f_full + alpha * hemi_side

        with torch.no_grad():
            full_norm = f_full.norm(dim=-1).mean()
            left_norm = f_left.norm(dim=-1).mean()
            right_norm = f_right.norm(dim=-1).mean()
            side_norm = hemi_side.norm(dim=-1).mean()
            fused_norm = f_fused.norm(dim=-1).mean()
            lr_cos = F.cosine_similarity(f_left, f_right, dim=-1).mean()
            self.last_stats = {
                "alpha": float(alpha.detach().cpu().item()),
                "full_norm_mean": float(full_norm.detach().cpu().item()),
                "left_norm_mean": float(left_norm.detach().cpu().item()),
                "right_norm_mean": float(right_norm.detach().cpu().item()),
                "side_norm_mean": float(side_norm.detach().cpu().item()),
                "fused_norm_mean": float(fused_norm.detach().cpu().item()),
                "left_right_cos_mean": float(lr_cos.detach().cpu().item()),
            }
        return f_fused
