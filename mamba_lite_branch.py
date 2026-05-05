import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _bounded_raw(target_value, max_value):
    eps = 1e-6
    ratio = min(max(float(target_value) / max(float(max_value), eps), eps), 1.0 - eps)
    return torch.log(torch.tensor(ratio / (1.0 - ratio)))


class SelectiveSSMLiteBlock(nn.Module):
    """
    Pure-PyTorch Mamba-inspired block.

    It keeps the implementation Windows-friendly: depthwise temporal mixing +
    input-dependent decay/update gates + a lightweight recurrent selective scan.
    """

    def __init__(self, d_model=128, kernel_size=3, dropout=0.1):
        super(SelectiveSSMLiteBlock, self).__init__()
        self.d_model = int(d_model)
        padding = int(kernel_size) // 2
        self.norm = nn.LayerNorm(self.d_model)
        self.in_proj = nn.Linear(self.d_model, self.d_model * 2)
        self.dwconv = nn.Conv1d(
            self.d_model,
            self.d_model,
            kernel_size=int(kernel_size),
            padding=padding,
            groups=self.d_model,
        )
        self.delta_proj = nn.Linear(self.d_model, self.d_model)
        self.value_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(dropout)
        self.a_log = nn.Parameter(torch.zeros(self.d_model))
        self.last_state_norm_mean = 0.0

    def _scan(self, conv_feat, gate):
        bsz, steps, dim = conv_feat.shape
        delta = torch.sigmoid(self.delta_proj(conv_feat))
        value = self.value_proj(conv_feat)
        decay_base = -F.softplus(self.a_log).view(1, dim)
        state = conv_feat.new_zeros(bsz, dim)
        outputs = []
        for t in range(steps):
            decay = torch.exp(delta[:, t, :] * decay_base)
            state = decay * state + (1.0 - decay) * value[:, t, :]
            outputs.append(state * torch.sigmoid(gate[:, t, :]))
        out = torch.stack(outputs, dim=1)
        with torch.no_grad():
            self.last_state_norm_mean = float(state.norm(dim=-1).mean().detach().cpu().item())
        return out

    def _directional_forward(self, x):
        mixed, gate = self.in_proj(x).chunk(2, dim=-1)
        mixed = self.dwconv(mixed.transpose(1, 2)).transpose(1, 2)
        mixed = F.silu(mixed)
        return self._scan(mixed, gate)

    def forward(self, x):
        residual = x
        x_norm = self.norm(x)
        out_fwd = self._directional_forward(x_norm)
        out_bwd = torch.flip(self._directional_forward(torch.flip(x_norm, dims=[1])), dims=[1])
        out = 0.5 * (out_fwd + out_bwd)
        out = self.dropout(self.out_proj(out))
        return residual + out


class MambaLiteFusionBranch(nn.Module):
    """
    Parallel Bi-SSM branch fused into the original DMMR encoder feature.

    Input sequence stays [B,T,310]. Output feature stays [B,64], so DMMR
    decoder/discriminator/classifier interfaces remain unchanged.
    """

    def __init__(
        self,
        input_dim=310,
        feature_dim=64,
        d_model=128,
        num_layers=1,
        kernel_size=3,
        dropout=0.1,
        alpha_init=0.2,
        alpha_max=0.8,
        delta_init_std=0.01,
    ):
        super(MambaLiteFusionBranch, self).__init__()
        self.feature_dim = int(feature_dim)
        self.alpha_max = float(alpha_max)
        self.input_proj = nn.Linear(int(input_dim), int(d_model))
        self.blocks = nn.ModuleList(
            [
                SelectiveSSMLiteBlock(
                    d_model=int(d_model),
                    kernel_size=int(kernel_size),
                    dropout=float(dropout),
                )
                for _ in range(int(num_layers))
            ]
        )
        self.temporal_pool = nn.Sequential(
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

    def forward(self, x, base_feat):
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        scores = self.temporal_pool(h).squeeze(-1)
        attn = F.softmax(scores, dim=-1)
        pooled = torch.sum(h * attn.unsqueeze(-1), dim=1)
        mamba_feat = self.to_feature(pooled)
        fusion_input = torch.cat(
            [base_feat, mamba_feat, base_feat - mamba_feat, torch.abs(base_feat - mamba_feat)],
            dim=-1,
        )
        delta = self.fusion(fusion_input)
        alpha = self.get_alpha()
        fused = base_feat + alpha * delta
        with torch.no_grad():
            safe_attn = attn.clamp_min(1e-12)
            entropy = -(safe_attn * safe_attn.log()).sum(dim=-1)
            self.last_stats = {
                "alpha": float(alpha.detach().cpu().item()),
                "base_feat_norm_mean": float(base_feat.norm(dim=-1).mean().detach().cpu().item()),
                "mamba_feat_norm_mean": float(mamba_feat.norm(dim=-1).mean().detach().cpu().item()),
                "delta_norm_mean": float(delta.norm(dim=-1).mean().detach().cpu().item()),
                "feature_delta_norm_mean": float((fused - base_feat).norm(dim=-1).mean().detach().cpu().item()),
                "attn_entropy_norm_mean": float((entropy / math.log(float(max(attn.shape[-1], 2)))).mean().detach().cpu().item()),
                "attn_max_mean": float(attn.max(dim=-1).values.mean().detach().cpu().item()),
                "last_state_norm_mean": float(sum(block.last_state_norm_mean for block in self.blocks) / max(len(self.blocks), 1)),
                "has_nan_or_inf": bool(
                    torch.isnan(fused).any().item()
                    or torch.isinf(fused).any().item()
                    or torch.isnan(attn).any().item()
                    or torch.isinf(attn).any().item()
                ),
            }
        return fused
