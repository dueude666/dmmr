import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReliabilitySourcePrototypeAttention(nn.Module):
    """Feature-level source-class prototype attention with a weak residual gate."""

    def __init__(
        self,
        feature_dim=64,
        num_subjects=14,
        num_classes=3,
        temperature=0.2,
        alpha_init=0.1,
        alpha_max=0.5,
        reliability_tau=1.0,
        reliability_min=0.8,
        reliability_max=1.2,
        hidden_dim=128,
        dropout=0.1,
        use_warmup=False,
        warmup_epochs=2,
        ramp_epochs=4,
        use_class_hint=False,
        class_hint_weight=1.0,
        class_hint_detach=True,
        filter_low_conf=False,
        min_reliability=0.0,
        source_balance=False,
        source_cap=0.12,
        adaptive_gate=False,
        adaptive_gate_min=0.0,
        adaptive_gate_max=1.0,
        centered_adaptive_gate=False,
        centered_gate_delta=0.2,
        gate_output_init_std=0.0,
    ):
        super(ReliabilitySourcePrototypeAttention, self).__init__()
        self.feature_dim = int(feature_dim)
        self.num_subjects = int(num_subjects)
        self.num_classes = int(num_classes)
        self.temperature = float(temperature)
        self.alpha_max = float(alpha_max)
        self.reliability_tau = float(reliability_tau)
        self.reliability_min = float(reliability_min)
        self.reliability_max = float(reliability_max)
        self.use_warmup = bool(use_warmup)
        self.warmup_epochs = int(warmup_epochs)
        self.ramp_epochs = max(1, int(ramp_epochs))
        self.use_class_hint = bool(use_class_hint)
        self.class_hint_weight = float(class_hint_weight)
        self.class_hint_detach = bool(class_hint_detach)
        self.filter_low_conf = bool(filter_low_conf)
        self.min_reliability = float(min_reliability)
        self.source_balance = bool(source_balance)
        self.source_cap = float(source_cap)
        self.adaptive_gate = bool(adaptive_gate)
        self.adaptive_gate_min = float(adaptive_gate_min)
        self.adaptive_gate_max = float(adaptive_gate_max)
        self.centered_adaptive_gate = bool(centered_adaptive_gate)
        self.centered_gate_delta = float(centered_gate_delta)
        self.gate_output_init_std = float(gate_output_init_std)

        self.register_buffer(
            "source_class_prototypes",
            torch.zeros(self.num_subjects, self.num_classes, self.feature_dim),
            persistent=True,
        )
        self.register_buffer(
            "source_class_counts",
            torch.zeros(self.num_subjects, self.num_classes),
            persistent=True,
        )
        self.register_buffer(
            "global_class_prototypes",
            torch.zeros(self.num_classes, self.feature_dim),
            persistent=True,
        )

        self.q_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.k_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.v_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.delta = nn.Sequential(
            nn.Linear(self.feature_dim * 2, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.feature_dim),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(self.feature_dim * 3, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)
        if self.gate_output_init_std > 0:
            nn.init.normal_(self.gate_net[-1].weight, mean=0.0, std=self.gate_output_init_std)
        else:
            nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.zeros_(self.gate_net[-1].bias)

        self.alpha_raw = nn.Parameter(torch.tensor(0.0))
        self._set_alpha_init(alpha_init)
        self.last_stats = {}

    def _set_alpha_init(self, alpha_init):
        eps = 1e-6
        ratio = min(max(float(alpha_init) / max(self.alpha_max, eps), eps), 1.0 - eps)
        raw = math.log(ratio / (1.0 - ratio))
        with torch.no_grad():
            self.alpha_raw.copy_(torch.tensor(raw, dtype=self.alpha_raw.dtype))

    def get_alpha_value(self):
        return float((self.alpha_max * torch.sigmoid(self.alpha_raw)).detach().cpu().item())

    def get_last_stats(self):
        return dict(self.last_stats)

    def get_inject_scale(self, current_epoch=None):
        if not self.use_warmup or current_epoch is None:
            return 1.0
        current_epoch = int(current_epoch)
        if current_epoch < self.warmup_epochs:
            return 0.0
        if current_epoch < self.warmup_epochs + self.ramp_epochs:
            return float(current_epoch - self.warmup_epochs + 1) / float(self.ramp_epochs)
        return 1.0

    def set_prototypes(self, prototypes, counts):
        expected = (self.num_subjects, self.num_classes, self.feature_dim)
        if tuple(prototypes.shape) != expected:
            raise ValueError("RSPA prototype shape {} does not match {}".format(tuple(prototypes.shape), expected))
        if tuple(counts.shape) != (self.num_subjects, self.num_classes):
            raise ValueError("RSPA counts shape {} does not match {}".format(tuple(counts.shape), (self.num_subjects, self.num_classes)))
        with torch.no_grad():
            prototypes = prototypes.to(self.source_class_prototypes.device, dtype=self.source_class_prototypes.dtype)
            counts = counts.to(self.source_class_counts.device, dtype=self.source_class_counts.dtype)
            self.source_class_prototypes.copy_(prototypes)
            self.source_class_counts.copy_(counts)
            class_counts = counts.sum(dim=0).clamp_min(1.0).unsqueeze(-1)
            global_proto = (prototypes * counts.unsqueeze(-1)).sum(dim=0) / class_counts
            self.global_class_prototypes.copy_(global_proto)

    def _prototype_reliability(self):
        valid = self.source_class_counts > 0
        if not valid.any():
            return torch.zeros_like(self.source_class_counts), valid
        diff = self.source_class_prototypes - self.global_class_prototypes.unsqueeze(0)
        dist = torch.norm(diff, dim=-1, p=2)
        reliability = torch.exp(-dist / max(self.reliability_tau, 1e-6))
        reliability = reliability.clamp(self.reliability_min, self.reliability_max)
        reliability = torch.where(valid, reliability, torch.zeros_like(reliability))
        if self.filter_low_conf:
            valid = valid & (reliability >= self.min_reliability)
            reliability = torch.where(valid, reliability, torch.zeros_like(reliability))
        return reliability, valid

    def forward(self, feat, current_epoch=None, class_probs=None):
        reliability, valid = self._prototype_reliability()
        flat_proto = self.source_class_prototypes.view(-1, self.feature_dim)
        flat_rel = reliability.view(-1)
        flat_valid = valid.view(-1)

        if int(flat_valid.sum().item()) < 2:
            self.last_stats = {
                "alpha": self.get_alpha_value(),
                "valid_prototypes": int(flat_valid.sum().item()),
                "attn_entropy_norm_mean": 0.0,
                "attn_max_mean": 0.0,
                "reliability_mean": 0.0,
                "reliability_min": 0.0,
                "reliability_max": 0.0,
                "context_norm_mean": 0.0,
                "delta_norm_mean": 0.0,
                "feature_delta_norm_mean": 0.0,
                "inject_scale": self.get_inject_scale(current_epoch),
                "use_class_hint": self.use_class_hint,
                "class_hint_weight": self.class_hint_weight,
                "class_hint_conf_mean": 0.0,
                "filter_low_conf": self.filter_low_conf,
                "min_reliability": self.min_reliability,
                "source_balance": self.source_balance,
                "source_cap": self.source_cap,
                "source_mass_max_mean": 0.0,
                "adaptive_gate": self.adaptive_gate,
                "centered_adaptive_gate": self.centered_adaptive_gate,
                "gate_output_init_std": self.gate_output_init_std,
                "adaptive_gate_mean": 1.0,
                "adaptive_gate_min_value": 1.0,
                "adaptive_gate_max_value": 1.0,
                "has_nan_or_inf": False,
            }
            return feat

        q = F.normalize(self.q_proj(feat), p=2, dim=-1)
        k = F.normalize(self.k_proj(flat_proto), p=2, dim=-1)
        v = self.v_proj(flat_proto)
        logits = torch.matmul(q, k.transpose(0, 1)) / max(self.temperature, 1e-6)
        logits = logits + flat_rel.clamp_min(1e-6).log().unsqueeze(0)
        class_hint_conf_mean = 0.0
        if self.use_class_hint and class_probs is not None:
            if class_probs.shape[-1] != self.num_classes:
                raise ValueError("RSPA class_probs last dim {} does not match num_classes {}".format(class_probs.shape[-1], self.num_classes))
            probs = class_probs.detach() if self.class_hint_detach else class_probs
            flat_class_ids = torch.arange(self.num_classes, device=feat.device).repeat(self.num_subjects)
            class_hint = probs[:, flat_class_ids].clamp_min(1e-6)
            logits = logits + self.class_hint_weight * class_hint.log()
            with torch.no_grad():
                class_hint_conf_mean = float(probs.max(dim=-1).values.mean().detach().cpu().item())
        logits = logits.masked_fill(~flat_valid.unsqueeze(0), -1e4)
        attn = F.softmax(logits, dim=-1)
        source_mass_max_mean = 0.0
        if self.source_balance:
            attn_3d = attn.view(attn.shape[0], self.num_subjects, self.num_classes)
            source_mass = attn_3d.sum(dim=-1, keepdim=True)
            cap = max(self.source_cap, 1e-6)
            scale = torch.clamp(cap / source_mass.clamp_min(1e-12), max=1.0)
            attn_3d = attn_3d * scale
            attn = attn_3d.view(attn.shape[0], -1)
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            with torch.no_grad():
                source_mass_max_mean = float(source_mass.squeeze(-1).max(dim=-1).values.mean().detach().cpu().item())
        context = torch.matmul(attn, v)
        delta = self.delta(torch.cat([feat, context], dim=-1))
        alpha = self.alpha_max * torch.sigmoid(self.alpha_raw)
        inject_scale = self.get_inject_scale(current_epoch)
        sample_gate = torch.ones(feat.shape[0], 1, device=feat.device, dtype=feat.dtype)
        if self.centered_adaptive_gate:
            gate_in = torch.cat([feat, context, torch.abs(feat - context)], dim=-1)
            sample_gate = 1.0 + self.centered_gate_delta * torch.tanh(self.gate_net(gate_in))
        elif self.adaptive_gate:
            gate_in = torch.cat([feat, context, torch.abs(feat - context)], dim=-1)
            raw_gate = torch.sigmoid(self.gate_net(gate_in))
            gate_range = max(self.adaptive_gate_max - self.adaptive_gate_min, 1e-6)
            sample_gate = self.adaptive_gate_min + gate_range * raw_gate
        out = feat + float(inject_scale) * alpha * sample_gate * delta

        with torch.no_grad():
            entropy = -(attn.clamp_min(1e-12) * attn.clamp_min(1e-12).log()).sum(dim=-1)
            denom = math.log(max(int(flat_valid.sum().item()), 2))
            rel_valid = flat_rel[flat_valid]
            self.last_stats = {
                "alpha": self.get_alpha_value(),
                "valid_prototypes": int(flat_valid.sum().item()),
                "filtered_prototypes": int((self.source_class_counts > 0).sum().item() - flat_valid.sum().item()),
                "attn_entropy_norm_mean": float((entropy / denom).mean().detach().cpu().item()),
                "attn_max_mean": float(attn.max(dim=-1).values.mean().detach().cpu().item()),
                "reliability_mean": float(rel_valid.mean().detach().cpu().item()),
                "reliability_min": float(rel_valid.min().detach().cpu().item()),
                "reliability_max": float(rel_valid.max().detach().cpu().item()),
                "context_norm_mean": float(context.norm(dim=-1).mean().detach().cpu().item()),
                "delta_norm_mean": float(delta.norm(dim=-1).mean().detach().cpu().item()),
                "feature_delta_norm_mean": float((out - feat).norm(dim=-1).mean().detach().cpu().item()),
                "inject_scale": float(inject_scale),
                "use_class_hint": self.use_class_hint,
                "class_hint_weight": self.class_hint_weight,
                "class_hint_conf_mean": class_hint_conf_mean,
                "filter_low_conf": self.filter_low_conf,
                "min_reliability": self.min_reliability,
                "source_balance": self.source_balance,
                "source_cap": self.source_cap,
                "source_mass_max_mean": source_mass_max_mean,
                "adaptive_gate": self.adaptive_gate,
                "centered_adaptive_gate": self.centered_adaptive_gate,
                "centered_gate_delta": self.centered_gate_delta,
                "gate_output_init_std": self.gate_output_init_std,
                "adaptive_gate_mean": float(sample_gate.mean().detach().cpu().item()),
                "adaptive_gate_min_value": float(sample_gate.min().detach().cpu().item()),
                "adaptive_gate_max_value": float(sample_gate.max().detach().cpu().item()),
                "has_nan_or_inf": bool(
                    torch.isnan(out).any().item()
                    or torch.isinf(out).any().item()
                    or torch.isnan(attn).any().item()
                    or torch.isinf(attn).any().item()
                ),
            }
        return out
