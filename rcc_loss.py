import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RCCLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        num_subjects,
        feature_dim,
        rcc_lambda=0.05,
        tau=0.2,
        reliability_tau=0.5,
        warmup_epochs=10,
        ramp_epochs=10,
        ema_momentum=0.9,
        reliability_min=0.5,
        reliability_max=1.5,
        min_valid_samples=4,
        use_reliability=True,
        update_centers=True,
    ):
        super(RCCLoss, self).__init__()
        self.num_classes = int(num_classes)
        self.num_subjects = int(num_subjects)
        self.feature_dim = int(feature_dim)
        self.rcc_lambda = float(rcc_lambda)
        self.tau = float(tau)
        self.reliability_tau = float(reliability_tau)
        self.warmup_epochs = int(warmup_epochs)
        self.ramp_epochs = max(1, int(ramp_epochs))
        self.ema_momentum = float(ema_momentum)
        self.reliability_min = float(reliability_min)
        self.reliability_max = float(reliability_max)
        self.min_valid_samples = int(min_valid_samples)
        self.use_reliability = bool(use_reliability)
        self.enable_center_updates = bool(update_centers)

        self.register_buffer("class_centers", torch.zeros(self.num_classes, self.feature_dim))
        self.register_buffer("class_initialized", torch.zeros(self.num_classes, dtype=torch.bool))
        self.register_buffer(
            "subject_class_centers",
            torch.zeros(self.num_subjects, self.num_classes, self.feature_dim),
        )
        self.register_buffer(
            "subject_class_initialized",
            torch.zeros(self.num_subjects, self.num_classes, dtype=torch.bool),
        )

    def _lambda_now(self, epoch):
        if epoch is None:
            return 0.0
        epoch = int(epoch)
        if epoch < self.warmup_epochs:
            return 0.0
        ramp = min(1.0, float(epoch - self.warmup_epochs + 1) / float(self.ramp_epochs))
        return self.rcc_lambda * ramp

    def _safe_zero(self, features, lambda_now, warning=None):
        center_norm = 0.0
        if self.class_centers.numel() > 0:
            center_norm = float(self.class_centers.norm(dim=-1).mean().detach().cpu().item())
        stats = {
            "rcc_loss_raw": 0.0,
            "rcc_lambda_now": float(lambda_now),
            "rcc_loss_weighted": 0.0,
            "reliability_mean": 1.0,
            "reliability_min": 1.0,
            "reliability_max": 1.0,
            "center_norm_mean": center_norm,
            "valid_rcc_samples": 0,
            "warning": warning or "",
        }
        return features.new_zeros(()), stats

    @torch.no_grad()
    def _ema_update(self, z, labels, subject_ids):
        if not self.enable_center_updates:
            return
        for cls_idx in labels.unique().tolist():
            cls_idx = int(cls_idx)
            cls_mask = labels == cls_idx
            if cls_mask.any():
                batch_center = z[cls_mask].mean(dim=0)
                if self.class_initialized[cls_idx]:
                    self.class_centers[cls_idx] = (
                        self.ema_momentum * self.class_centers[cls_idx]
                        + (1.0 - self.ema_momentum) * batch_center
                    )
                else:
                    self.class_centers[cls_idx] = batch_center
                    self.class_initialized[cls_idx] = True
        pair_ids = torch.stack([subject_ids, labels], dim=1).unique(dim=0)
        for subj_idx, cls_idx in pair_ids.tolist():
            subj_idx = int(subj_idx)
            cls_idx = int(cls_idx)
            mask = (subject_ids == subj_idx) & (labels == cls_idx)
            if mask.any():
                batch_center = z[mask].mean(dim=0)
                if self.subject_class_initialized[subj_idx, cls_idx]:
                    self.subject_class_centers[subj_idx, cls_idx] = (
                        self.ema_momentum * self.subject_class_centers[subj_idx, cls_idx]
                        + (1.0 - self.ema_momentum) * batch_center
                    )
                else:
                    self.subject_class_centers[subj_idx, cls_idx] = batch_center
                    self.subject_class_initialized[subj_idx, cls_idx] = True

    @torch.no_grad()
    def update_centers(self, features, labels, subject_ids=None):
        if features is None or features.numel() == 0 or not self.enable_center_updates:
            return
        if subject_ids is None:
            return
        labels = labels.view(-1).long()
        subject_ids = subject_ids.view(-1).long()
        if features.shape[0] != labels.shape[0] or features.shape[0] != subject_ids.shape[0]:
            return
        z = F.normalize(features.detach(), dim=1)
        if not torch.isfinite(z).all():
            return
        self._ema_update(z, labels.detach(), subject_ids.detach())

    def forward(self, features, labels, subject_ids=None, epoch=None):
        lambda_now = self._lambda_now(epoch)
        if features is None or features.numel() == 0:
            return self._safe_zero(features if features is not None else torch.zeros(1), lambda_now, "empty_features")
        if lambda_now <= 0.0:
            return self._safe_zero(features, lambda_now, "warmup")

        labels = labels.view(-1).long()
        if subject_ids is None:
            return self._safe_zero(features, lambda_now, "missing_subject_ids")
        subject_ids = subject_ids.view(-1).long()
        if features.shape[0] != labels.shape[0] or features.shape[0] != subject_ids.shape[0]:
            return self._safe_zero(features, lambda_now, "shape_mismatch")

        z = F.normalize(features, dim=1)
        if not torch.isfinite(z).all():
            return self._safe_zero(features, lambda_now, "non_finite_features")

        class_centers = self.class_centers.detach().clone()
        class_valid = self.class_initialized.detach().clone()
        subject_class_centers = self.subject_class_centers.detach().clone()
        subject_class_valid = self.subject_class_initialized.detach().clone()
        num_valid_classes = int(class_valid.sum().item())
        if num_valid_classes < 2:
            return self._safe_zero(features, lambda_now, "insufficient_initialized_classes")

        logits = torch.matmul(z, class_centers.t()) / max(self.tau, 1e-6)
        invalid_mask = ~class_valid
        if invalid_mask.any():
            logits = logits.masked_fill(invalid_mask.unsqueeze(0), -1e4)

        valid_sample_mask = class_valid[labels]
        if valid_sample_mask.sum().item() < self.min_valid_samples:
            return self._safe_zero(features, lambda_now, "too_few_valid_samples")

        loss_center_all = F.cross_entropy(logits, labels, reduction="none")
        if not torch.isfinite(loss_center_all).all():
            return self._safe_zero(features, lambda_now, "non_finite_center_loss")

        reliability = torch.ones_like(loss_center_all)
        if self.use_reliability:
            valid_subject_class = subject_class_valid[subject_ids, labels]
            if valid_subject_class.any():
                subj_centers = subject_class_centers[subject_ids[valid_subject_class], labels[valid_subject_class]]
                cls_centers = class_centers[labels[valid_subject_class]]
                d_sc = torch.norm(subj_centers - cls_centers, dim=-1, p=2)
                rel = torch.exp(-d_sc / max(self.reliability_tau, 1e-6))
                rel = rel.clamp(self.reliability_min, self.reliability_max)
                reliability[valid_subject_class] = rel
            reliability[~valid_subject_class] = 1.0

        weighted_losses = reliability * loss_center_all
        valid_weighted_losses = weighted_losses[valid_sample_mask]
        if valid_weighted_losses.numel() < self.min_valid_samples:
            return self._safe_zero(features, lambda_now, "too_few_valid_weighted_samples")

        loss_rcc = valid_weighted_losses.mean()
        if not torch.isfinite(loss_rcc):
            return self._safe_zero(features, lambda_now, "non_finite_rcc_loss")

        reliability_valid = reliability[valid_sample_mask]
        center_norms = class_centers[class_valid].norm(dim=-1) if class_valid.any() else class_centers.new_zeros(1)
        stats = {
            "rcc_loss_raw": float(loss_rcc.detach().cpu().item()),
            "rcc_lambda_now": float(lambda_now),
            "rcc_loss_weighted": float((lambda_now * loss_rcc).detach().cpu().item()),
            "reliability_mean": float(reliability_valid.mean().detach().cpu().item()),
            "reliability_min": float(reliability_valid.min().detach().cpu().item()),
            "reliability_max": float(reliability_valid.max().detach().cpu().item()),
            "center_norm_mean": float(center_norms.mean().detach().cpu().item()),
            "valid_rcc_samples": int(valid_sample_mask.sum().item()),
            "warning": "",
        }
        return loss_rcc, stats
