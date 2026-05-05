import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SourceSubjectPromptBank(nn.Module):
    def __init__(
        self,
        feature_dim,
        num_subjects_total,
        prompt_tau=2.0,
        prompt_alpha_max=0.2,
        prompt_beta_max=0.3,
        prompt_alpha_init=0.1,
        prompt_beta_init=0.1,
        prompt_dropout=0.0,
        use_zero_init_prompt_residual=True,
        prompt_fusion_dropout=0.1,
        prompt_gate_init=0.01,
        use_prompt_gate_warmup=False,
        prompt_warmup_epochs=2,
        prompt_ramp_epochs=2,
        use_prompt_fusion_detach=False,
    ):
        super(SourceSubjectPromptBank, self).__init__()
        self.feature_dim = feature_dim
        self.num_subjects_total = num_subjects_total
        self.prompt_tau = float(prompt_tau)
        self.prompt_alpha_max = float(prompt_alpha_max)
        self.prompt_beta_max = float(prompt_beta_max)
        self.use_zero_init_prompt_residual = bool(use_zero_init_prompt_residual)
        self.use_prompt_gate_warmup = bool(use_prompt_gate_warmup)
        self.prompt_warmup_epochs = int(prompt_warmup_epochs)
        self.prompt_ramp_epochs = max(1, int(prompt_ramp_epochs))
        self.use_prompt_fusion_detach = bool(use_prompt_fusion_detach)
        self.prompt_gate = nn.Parameter(torch.tensor(float(prompt_gate_init), dtype=torch.float32))

        self.prompt_bank = nn.Parameter(torch.randn(num_subjects_total, feature_dim) * 0.02)
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(prompt_dropout)
        self.alpha_raw = nn.Parameter(torch.tensor(self._raw_from_init(prompt_alpha_init, self.prompt_alpha_max), dtype=torch.float32))
        self.beta_raw = nn.Parameter(torch.tensor(self._raw_from_init(prompt_beta_init, self.prompt_beta_max), dtype=torch.float32))
        self.fusion_net = nn.Sequential(
            nn.Linear(feature_dim + feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(prompt_fusion_dropout),
            nn.Linear(feature_dim, feature_dim),
        )

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.xavier_uniform_(self.fusion_net[0].weight)
        nn.init.zeros_(self.fusion_net[0].bias)
        nn.init.xavier_uniform_(self.fusion_net[-1].weight)
        nn.init.zeros_(self.fusion_net[-1].bias)
        self.last_attn_mean = None
        self.last_top3_indices = None
        self.last_fusion_out_norm = None
        self.last_feature_delta_norm = None

    @staticmethod
    def _raw_from_init(init_value, max_value):
        ratio = float(init_value) / float(max_value)
        ratio = min(max(ratio, 1e-4), 1.0 - 1e-4)
        return math.log(ratio / (1.0 - ratio))

    def get_alpha_value(self):
        alpha = self.prompt_alpha_max * torch.sigmoid(self.alpha_raw)
        return float(alpha.detach().cpu().item())

    def get_beta_value(self):
        beta = self.prompt_beta_max * torch.sigmoid(self.beta_raw)
        return float(beta.detach().cpu().item())

    def get_last_attention_debug(self):
        return {
            "attn_mean": self.last_attn_mean,
            "top3_prompt_indices": self.last_top3_indices,
            "fusion_out_norm": self.last_fusion_out_norm,
            "feature_delta_norm": self.last_feature_delta_norm,
        }

    def get_prompt_gate_value(self):
        return float(self.prompt_gate.detach().cpu().item())

    def get_inject_scale(self, current_epoch):
        if (not self.use_prompt_gate_warmup) or (current_epoch is None):
            return 1.0
        if current_epoch < self.prompt_warmup_epochs:
            return 0.0
        if current_epoch < self.prompt_warmup_epochs + self.prompt_ramp_epochs:
            return float(current_epoch - self.prompt_warmup_epochs + 1) / float(self.prompt_ramp_epochs)
        return 1.0

    def is_fusion_last_zero_initialized(self):
        w = self.fusion_net[-1].weight.detach()
        b = self.fusion_net[-1].bias.detach()
        return bool(torch.allclose(w, torch.zeros_like(w)) and torch.allclose(b, torch.zeros_like(b)))

    def forward(self, z, source_prompt_mask, current_epoch=None):
        if z.dim() != 2:
            raise ValueError("SourceSubjectPromptBank expects z as [B, D], got shape {}".format(tuple(z.shape)))
        if z.size(1) != self.feature_dim:
            raise ValueError("SourceSubjectPromptBank expects feature dim {}, got {}".format(self.feature_dim, z.size(1)))

        if source_prompt_mask is None:
            source_prompt_mask = torch.ones(self.num_subjects_total, dtype=torch.bool, device=z.device)
        else:
            source_prompt_mask = source_prompt_mask.to(device=z.device, dtype=torch.bool)
        if source_prompt_mask.numel() != self.num_subjects_total:
            raise ValueError("Source prompt mask length {}, expected {}".format(source_prompt_mask.numel(), self.num_subjects_total))
        if not torch.any(source_prompt_mask):
            source_prompt_mask = torch.ones_like(source_prompt_mask)

        source_indices = torch.nonzero(source_prompt_mask, as_tuple=False).squeeze(1)
        source_prompts = self.prompt_bank[source_prompt_mask]
        p_global = source_prompts.mean(dim=0)
        residual_prompts = source_prompts - p_global.unsqueeze(0)

        q = self.q_proj(z)                              # [B, D]
        k = self.k_proj(residual_prompts)               # [Ns, D]
        v = self.v_proj(residual_prompts)               # [Ns, D]
        attn_logits = torch.matmul(q, k.transpose(0, 1)) / math.sqrt(self.feature_dim)
        attn = F.softmax(attn_logits / self.prompt_tau, dim=-1)  # [B, Ns]
        residual_context = torch.matmul(attn, v)        # [B, D]
        beta = self.prompt_beta_max * torch.sigmoid(self.beta_raw)
        context = p_global.unsqueeze(0) + beta * residual_context
        context = self.dropout(context)
        alpha = self.prompt_alpha_max * torch.sigmoid(self.alpha_raw)
        gated_prompt_feature = alpha * context
        if self.use_prompt_fusion_detach:
            concat_feature = torch.cat([z.detach(), gated_prompt_feature], dim=-1)
        else:
            concat_feature = torch.cat([z, gated_prompt_feature], dim=-1)
        fusion_out = self.fusion_net(concat_feature)
        inject_scale = self.get_inject_scale(current_epoch)
        out = z + inject_scale * self.prompt_gate * fusion_out

        with torch.no_grad():
            attn_mean = attn.mean(dim=0)
            k_top = min(3, attn_mean.numel())
            _, top_local = torch.topk(attn_mean, k=k_top, dim=0)
            top_global = source_indices[top_local]
            self.last_attn_mean = float(attn_mean.mean().detach().cpu().item())
            self.last_top3_indices = [int(x) for x in top_global.detach().cpu().tolist()]
            self.last_fusion_out_norm = float(fusion_out.norm(dim=-1).mean().detach().cpu().item())
            self.last_feature_delta_norm = float((out - z).norm(dim=-1).mean().detach().cpu().item())
        return out, attn
