import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassPrototypeCalibrator(nn.Module):
    """Add fixed source class prototype cosine logits to classifier logits."""

    def __init__(
        self,
        feature_dim=64,
        num_classes=3,
        alpha=0.1,
        temperature=0.2,
        learnable_alpha=True,
    ):
        super(ClassPrototypeCalibrator, self).__init__()
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.temperature = float(temperature)
        self.learnable_alpha = bool(learnable_alpha)
        self.register_buffer("class_prototypes", torch.zeros(self.num_classes, self.feature_dim), persistent=True)
        self.register_buffer("prototype_counts", torch.zeros(self.num_classes), persistent=True)
        if self.learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        else:
            self.register_buffer("alpha", torch.tensor(float(alpha)), persistent=True)
        self.last_stats = {}

    def set_prototypes(self, prototypes, counts):
        if prototypes.shape != self.class_prototypes.shape:
            raise ValueError(
                "prototype shape {} does not match {}".format(
                    tuple(prototypes.shape), tuple(self.class_prototypes.shape)
                )
            )
        if counts.shape[0] != self.num_classes:
            raise ValueError("counts shape {} does not match num_classes {}".format(tuple(counts.shape), self.num_classes))
        with torch.no_grad():
            self.class_prototypes.copy_(prototypes.to(self.class_prototypes.device, dtype=self.class_prototypes.dtype))
            self.prototype_counts.copy_(counts.to(self.prototype_counts.device, dtype=self.prototype_counts.dtype))

    def get_alpha_value(self):
        return float(self.alpha.detach().cpu().item())

    def get_last_stats(self):
        return dict(self.last_stats)

    def forward(self, feat, logits):
        feat_norm = F.normalize(feat, p=2, dim=-1)
        proto_norm = F.normalize(self.class_prototypes, p=2, dim=-1)
        temp = max(self.temperature, 1e-6)
        proto_logits = torch.matmul(feat_norm, proto_norm.transpose(0, 1)) / temp
        calibrated_logits = logits + self.alpha * proto_logits

        with torch.no_grad():
            probs = F.softmax(proto_logits, dim=-1)
            entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)
            self.last_stats = {
                "alpha": self.get_alpha_value(),
                "temperature": self.temperature,
                "proto_logit_mean": float(proto_logits.mean().detach().cpu().item()),
                "proto_logit_std": float(proto_logits.std(unbiased=False).detach().cpu().item()),
                "proto_entropy_mean": float(entropy.mean().detach().cpu().item()),
                "proto_entropy_norm_mean": float((entropy / math.log(max(self.num_classes, 2))).mean().detach().cpu().item()),
                "prototype_norm_mean": float(self.class_prototypes.norm(dim=-1).mean().detach().cpu().item()),
                "prototype_counts": [int(v) for v in self.prototype_counts.detach().cpu().tolist()],
                "has_nan_or_inf": bool(
                    torch.isnan(calibrated_logits).any().item()
                    or torch.isinf(calibrated_logits).any().item()
                    or torch.isnan(proto_logits).any().item()
                    or torch.isinf(proto_logits).any().item()
                ),
            }
        return calibrated_logits, proto_logits
