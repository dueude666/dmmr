import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _bounded_raw(target_value, max_value):
    eps = 1e-6
    ratio = min(max(float(target_value) / max(float(max_value), eps), eps), 1.0 - eps)
    return torch.log(torch.tensor(ratio / (1.0 - ratio)))


class PatchTransformerFusionBranch(nn.Module):
    """
    PatchTST/TimesNet-inspired temporal patch branch.

    It treats the ABP output as a multivariate time series, builds overlapping
    temporal patches, models patch tokens with a small Transformer, and injects
    a gated residual into the original DMMR [B,64] feature.
    """

    def __init__(
        self,
        input_dim=310,
        feature_dim=64,
        time_steps=30,
        patch_len=6,
        patch_stride=3,
        d_model=128,
        num_heads=4,
        num_layers=1,
        dropout=0.1,
        alpha_init=0.25,
        alpha_max=0.8,
        delta_init_std=0.02,
    ):
        super(PatchTransformerFusionBranch, self).__init__()
        self.input_dim = int(input_dim)
        self.feature_dim = int(feature_dim)
        self.time_steps = int(time_steps)
        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride)
        self.d_model = int(d_model)
        self.alpha_max = float(alpha_max)
        if self.patch_len <= 0 or self.patch_stride <= 0:
            raise ValueError("patch_len and patch_stride must be positive")
        self.num_patches = max(1, (self.time_steps - self.patch_len) // self.patch_stride + 1)

        self.patch_proj = nn.Sequential(
            nn.LayerNorm(self.patch_len * self.input_dim),
            nn.Linear(self.patch_len * self.input_dim, self.d_model),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.d_model))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(num_heads),
            dim_feedforward=self.d_model * 2,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.patch_score = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.Tanh(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_model // 2, 1),
        )
        self.to_feature = nn.Linear(self.d_model, self.feature_dim)
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

    def _patchify(self, x):
        bsz, steps, dim = x.shape
        if steps < self.patch_len:
            pad = self.patch_len - steps
            x = F.pad(x, (0, 0, 0, pad))
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.patch_stride)
        # unfold gives [B,num_patches,D,patch_len]; move patch_len before D.
        patches = patches.permute(0, 1, 3, 2).contiguous()
        return patches.view(bsz, patches.shape[1], self.patch_len * dim)

    def forward(self, x, base_feat):
        patches = self._patchify(x)
        h = self.patch_proj(patches)
        if h.shape[1] != self.pos_embed.shape[1]:
            pos = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=h.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        else:
            pos = self.pos_embed
        h = self.transformer(h + pos)
        patch_logits = self.patch_score(h).squeeze(-1)
        patch_attn = F.softmax(patch_logits, dim=-1)
        pooled = torch.sum(h * patch_attn.unsqueeze(-1), dim=1)
        patch_feat = self.to_feature(pooled)

        fusion_input = torch.cat(
            [base_feat, patch_feat, base_feat - patch_feat, torch.abs(base_feat - patch_feat)],
            dim=-1,
        )
        delta = self.fusion(fusion_input)
        alpha = self.get_alpha()
        fused = base_feat + alpha * delta
        with torch.no_grad():
            safe_attn = patch_attn.clamp_min(1e-12)
            entropy = -(safe_attn * safe_attn.log()).sum(dim=-1)
            self.last_stats = {
                "alpha": float(alpha.detach().cpu().item()),
                "base_feat_norm_mean": float(base_feat.norm(dim=-1).mean().detach().cpu().item()),
                "patch_feat_norm_mean": float(patch_feat.norm(dim=-1).mean().detach().cpu().item()),
                "delta_norm_mean": float(delta.norm(dim=-1).mean().detach().cpu().item()),
                "feature_delta_norm_mean": float((fused - base_feat).norm(dim=-1).mean().detach().cpu().item()),
                "patch_attn_entropy_norm_mean": float((entropy / math.log(float(max(patch_attn.shape[-1], 2)))).mean().detach().cpu().item()),
                "patch_attn_max_mean": float(patch_attn.max(dim=-1).values.mean().detach().cpu().item()),
                "num_patches": int(patch_attn.shape[-1]),
                "has_nan_or_inf": bool(
                    torch.isnan(fused).any().item()
                    or torch.isinf(fused).any().item()
                    or torch.isnan(patch_attn).any().item()
                    or torch.isinf(patch_attn).any().item()
                ),
            }
        return fused
