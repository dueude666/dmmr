import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperbolicContrastiveHead(nn.Module):
    def __init__(self, input_dim=64, proj_dim=32, temperature=0.1, curvature=1.0):
        super(HyperbolicContrastiveHead, self).__init__()
        self.input_dim = int(input_dim)
        self.proj_dim = int(proj_dim)
        self.temperature = float(temperature)
        self.curvature = float(curvature)
        self.eps = 1e-6

        self.proj = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_dim, self.proj_dim),
        )

    def _project_to_ball(self, x):
        c = max(self.curvature, self.eps)
        sqrt_c = math.sqrt(c)
        max_norm = (1.0 - 1e-4) / sqrt_c
        norm = torch.norm(x, p=2, dim=-1, keepdim=True).clamp_min(self.eps)
        scale = torch.clamp(max_norm / norm, max=1.0)
        return x * scale

    def _poincare_distance(self, x):
        c = max(self.curvature, self.eps)
        x2 = (x * x).sum(dim=-1, keepdim=True)  # [B,1]
        diff2 = (x.unsqueeze(1) - x.unsqueeze(0)).pow(2).sum(dim=-1)  # [B,B]
        denom = (1.0 - c * x2) * (1.0 - c * x2).transpose(0, 1)
        denom = denom.clamp_min(self.eps)
        arg = 1.0 + 2.0 * c * diff2 / denom
        arg = arg.clamp_min(1.0 + self.eps)
        return torch.acosh(arg) / math.sqrt(c)

    def _supcon_loss(self, z_hyp, labels):
        labels = labels.view(-1)
        batch_size = labels.size(0)
        if batch_size < 2:
            return z_hyp.new_zeros(())

        dist = self._poincare_distance(z_hyp)
        logits = -dist / max(self.temperature, self.eps)

        eye = torch.eye(batch_size, device=z_hyp.device, dtype=torch.bool)
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & (~eye)
        valid_mask = pos_mask.sum(dim=1) > 0
        if not torch.any(valid_mask):
            return z_hyp.new_zeros(())

        logits = logits - logits.max(dim=1, keepdim=True).values
        exp_logits = torch.exp(logits) * (~eye).float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(self.eps))

        mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / pos_mask.float().sum(dim=1).clamp_min(1.0)
        loss = -mean_log_prob_pos[valid_mask].mean()
        if torch.isnan(loss) or torch.isinf(loss):
            return z_hyp.new_zeros(())
        return loss

    def forward(self, feat, labels):
        z_euc = self.proj(feat)
        z_hyp = self._project_to_ball(z_euc)
        loss = self._supcon_loss(z_hyp, labels)
        z_norm = torch.norm(z_hyp, p=2, dim=-1)
        stats = {
            "z_hyp_norm_mean": float(z_norm.mean().detach().cpu().item()),
            "z_hyp_norm_max": float(z_norm.max().detach().cpu().item()),
            "has_nan_or_inf": bool((~torch.isfinite(z_hyp)).any().detach().cpu().item()),
        }
        return z_hyp, loss, stats
