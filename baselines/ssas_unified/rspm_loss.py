import torch
import torch.nn as nn
import torch.nn.functional as F


class ReliabilitySourcePrototypeMemory(nn.Module):
    """Reliability-guided source prototype memory for SSAS bottleneck features."""

    def __init__(
        self,
        num_domains,
        num_classes,
        feature_dim,
        temperature=0.2,
        momentum=0.9,
        target_conf_threshold=0.7,
        reliability_tau=1.0,
        reliability_min=0.5,
        reliability_max=1.5,
        target_weight=0.3,
        eps=1e-6,
    ):
        super().__init__()
        self.num_domains = num_domains
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.momentum = momentum
        self.target_conf_threshold = target_conf_threshold
        self.reliability_tau = reliability_tau
        self.reliability_min = reliability_min
        self.reliability_max = reliability_max
        self.target_weight = target_weight
        self.eps = eps
        self.register_buffer("global_centers", torch.zeros(num_classes, feature_dim))
        self.register_buffer("domain_centers", torch.zeros(num_domains, num_classes, feature_dim))
        self.register_buffer("global_seen", torch.zeros(num_classes, dtype=torch.bool))
        self.register_buffer("domain_seen", torch.zeros(num_domains, num_classes, dtype=torch.bool))

    @torch.no_grad()
    def update_source_centers(self, source_features, source_labels):
        all_features = []
        all_labels = []
        for domain_idx, (features, labels) in enumerate(zip(source_features, source_labels)):
            labels = labels.view(-1).long()
            features_detached = F.normalize(features.detach(), dim=1)
            all_features.append(features_detached)
            all_labels.append(labels.detach())
            for cls in range(self.num_classes):
                mask = labels == cls
                if not mask.any():
                    continue
                center = F.normalize(features_detached[mask].mean(dim=0), dim=0)
                if self.domain_seen[domain_idx, cls]:
                    self.domain_centers[domain_idx, cls].mul_(self.momentum).add_(center * (1.0 - self.momentum))
                else:
                    self.domain_centers[domain_idx, cls].copy_(center)
                    self.domain_seen[domain_idx, cls] = True
                self.domain_centers[domain_idx, cls].copy_(F.normalize(self.domain_centers[domain_idx, cls], dim=0))

        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        for cls in range(self.num_classes):
            mask = all_labels == cls
            if not mask.any():
                continue
            center = F.normalize(all_features[mask].mean(dim=0), dim=0)
            if self.global_seen[cls]:
                self.global_centers[cls].mul_(self.momentum).add_(center * (1.0 - self.momentum))
            else:
                self.global_centers[cls].copy_(center)
                self.global_seen[cls] = True
            self.global_centers[cls].copy_(F.normalize(self.global_centers[cls], dim=0))

    def forward(self, source_features, source_labels, target_features, target_logits):
        self.update_source_centers(source_features, source_labels)
        valid_classes = self.global_seen
        if valid_classes.sum() < 2:
            zero = target_features.new_tensor(0.0)
            return zero, self._stats(zero)

        centers = F.normalize(self.global_centers[valid_classes].detach(), dim=1)
        class_ids = torch.arange(self.num_classes, device=target_features.device)[valid_classes]
        source_losses = []
        reliability_values = []
        valid_source_samples = 0

        for domain_idx, (features, labels) in enumerate(zip(source_features, source_labels)):
            labels = labels.view(-1).long()
            valid_mask = torch.isin(labels, class_ids)
            if not valid_mask.any():
                continue
            feats = F.normalize(features[valid_mask], dim=1)
            labs = labels[valid_mask]
            logits = feats @ centers.t() / self.temperature
            targets = self._remap_labels(labs, class_ids)
            per_sample = F.cross_entropy(logits, targets, reduction="none")
            weights = self._domain_reliability(domain_idx, labs).to(per_sample.device)
            source_losses.append((weights * per_sample).mean())
            reliability_values.append(weights.detach())
            valid_source_samples += int(valid_mask.sum().item())

        source_loss = torch.stack(source_losses).mean() if source_losses else target_features.new_tensor(0.0)
        target_loss, valid_target_samples, target_conf_mean = self._target_alignment_loss(target_features, target_logits, centers, class_ids)
        total = source_loss + self.target_weight * target_loss
        if not torch.isfinite(total):
            total = target_features.new_tensor(0.0)

        reliability = torch.cat(reliability_values) if reliability_values else None
        stats = self._stats(
            total,
            valid_source_samples=valid_source_samples,
            valid_target_samples=valid_target_samples,
            reliability_mean=float(reliability.mean().item()) if reliability is not None else 0.0,
            reliability_min=float(reliability.min().item()) if reliability is not None else 0.0,
            reliability_max=float(reliability.max().item()) if reliability is not None else 0.0,
            center_norm=float(self.global_centers[valid_classes].norm(dim=1).mean().item()),
            target_conf_mean=target_conf_mean,
        )
        return total, stats

    def _domain_reliability(self, domain_idx, labels):
        weights = []
        for label in labels.view(-1).long():
            cls = int(label.item())
            if not self.global_seen[cls] or not self.domain_seen[domain_idx, cls]:
                weights.append(torch.tensor(1.0, device=labels.device))
                continue
            dist = torch.norm(self.domain_centers[domain_idx, cls] - self.global_centers[cls], p=2)
            rel = torch.exp(-dist / max(self.reliability_tau, self.eps))
            rel = torch.clamp(rel, self.reliability_min, self.reliability_max)
            weights.append(rel.to(labels.device))
        return torch.stack(weights)

    def _target_alignment_loss(self, target_features, target_logits, centers, class_ids):
        probs = F.softmax(target_logits.detach(), dim=1)
        conf, pseudo = probs.max(dim=1)
        valid_mask = (conf >= self.target_conf_threshold) & torch.isin(pseudo, class_ids)
        if not valid_mask.any():
            return target_features.new_tensor(0.0), 0, float(conf.mean().item())
        feats = F.normalize(target_features[valid_mask], dim=1)
        logits = feats @ centers.t() / self.temperature
        targets = self._remap_labels(pseudo[valid_mask].long(), class_ids)
        return F.cross_entropy(logits, targets), int(valid_mask.sum().item()), float(conf[valid_mask].mean().item())

    @staticmethod
    def _remap_labels(labels, class_ids):
        remapped = torch.zeros_like(labels)
        for idx, cls in enumerate(class_ids):
            remapped[labels == cls] = idx
        return remapped

    @staticmethod
    def _stats(
        loss,
        valid_source_samples=0,
        valid_target_samples=0,
        reliability_mean=0.0,
        reliability_min=0.0,
        reliability_max=0.0,
        center_norm=0.0,
        target_conf_mean=0.0,
    ):
        return {
            "rspm_loss": float(loss.detach().item()),
            "rspm_valid_source_samples": int(valid_source_samples),
            "rspm_valid_target_samples": int(valid_target_samples),
            "rspm_reliability_mean": float(reliability_mean),
            "rspm_reliability_min": float(reliability_min),
            "rspm_reliability_max": float(reliability_max),
            "rspm_center_norm": float(center_norm),
            "rspm_target_conf_mean": float(target_conf_mean),
        }
