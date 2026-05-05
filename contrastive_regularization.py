import torch
import torch.nn as nn
import torch.nn.functional as F


class SubjectInvariantContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, use_proj_head=False, feature_dim=64):
        super(SubjectInvariantContrastiveLoss, self).__init__()
        self.temperature = float(temperature)
        self.use_proj_head = bool(use_proj_head)
        if self.use_proj_head:
            self.proj_head = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, feature_dim),
            )
        else:
            self.proj_head = None
        self.last_stats = {}

    def forward(self, features, labels, subject_ids=None):
        del subject_ids  # Reserved for future subject-aware positive/negative masks.
        labels = labels.view(-1)
        if self.proj_head is not None:
            features = self.proj_head(features)

        feature_norm_mean = features.detach().norm(dim=-1).mean()
        features = F.normalize(features, p=2, dim=-1)
        logits = torch.matmul(features, features.transpose(0, 1)) / max(self.temperature, 1e-8)
        logits = logits - logits.detach().max(dim=1, keepdim=True).values

        batch_size = labels.shape[0]
        eye = torch.eye(batch_size, dtype=torch.bool, device=features.device)
        label_equal = labels.unsqueeze(0).eq(labels.unsqueeze(1))
        positive_mask = label_equal & (~eye)
        valid_mask = positive_mask.sum(dim=1) > 0
        positive_pair_count = positive_mask.sum()

        if not torch.any(valid_mask):
            loss = features.new_zeros(())
        else:
            exp_logits = torch.exp(logits) * (~eye).float()
            log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
            mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1)
            loss = -mean_log_prob_pos[valid_mask].mean()

        self.last_stats = {
            "positive_pair_count": int(positive_pair_count.detach().cpu().item()),
            "valid_positive_ratio": float(valid_mask.float().mean().detach().cpu().item()),
            "feature_norm_mean": float(feature_norm_mean.detach().cpu().item()),
            "non_finite": int((~torch.isfinite(loss)).detach().cpu().item()),
        }
        if not torch.isfinite(loss):
            loss = features.new_zeros(())
        return loss, self.last_stats
