import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalGraphBlock(nn.Module):
    def __init__(
        self,
        num_channels=62,
        num_bands=5,
        dropout=0.1,
        kernel_size=3,
        alpha_init=0.1,
        use_gcn_residual=False,
        gcn_alpha_init=0.1,
        gcn_learnable_alpha=True,
        use_self_loop_prior=False,
        self_loop_weight=0.1,
        use_pre_lstm_dropout=False,
        pre_lstm_dropout_p=0.1,
        stable_adj_alpha=1.0,
    ):
        super(TemporalGraphBlock, self).__init__()
        self.num_channels = num_channels
        self.num_bands = num_bands
        self.feature_dim = num_channels * num_bands
        self.use_gcn_residual = use_gcn_residual
        self.use_self_loop_prior = use_self_loop_prior
        self.self_loop_weight = float(self_loop_weight)
        self.use_pre_lstm_dropout = use_pre_lstm_dropout
        self.stable_adj_alpha = float(stable_adj_alpha)

        self.A_logits = nn.Parameter(torch.zeros(num_channels, num_channels))
        self.band_mlp = nn.Linear(num_bands, num_bands)
        if gcn_learnable_alpha:
            self.gcn_alpha = nn.Parameter(torch.tensor(float(gcn_alpha_init)))
        else:
            self.register_buffer("gcn_alpha", torch.tensor(float(gcn_alpha_init)))

        padding = kernel_size // 2
        self.temporal_dw = nn.Conv1d(
            self.feature_dim,
            self.feature_dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=self.feature_dim,
        )
        self.temporal_pw = nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.feature_dim)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.pre_lstm_dropout = nn.Dropout(pre_lstm_dropout_p)

    def get_gcn_alpha_value(self):
        return float(self.gcn_alpha.detach().cpu().item())

    def get_adj_norm_stats(self):
        with torch.no_grad():
            adj_raw = 0.5 * (self.A_logits + self.A_logits.transpose(0, 1))
            if self.use_self_loop_prior:
                identity = torch.eye(self.num_channels, device=adj_raw.device, dtype=adj_raw.dtype)
                adj_raw = adj_raw + self.self_loop_weight * identity
            identity = torch.eye(self.num_channels, device=adj_raw.device, dtype=adj_raw.dtype)
            adj_stable = adj_raw * self.stable_adj_alpha + identity * (1.0 - self.stable_adj_alpha)
            a_norm = F.softmax(adj_stable, dim=-1)
            diag_mask = torch.eye(self.num_channels, device=a_norm.device, dtype=torch.bool)
            diag_mean = a_norm[diag_mask].mean().item()
            offdiag_mean = a_norm[~diag_mask].mean().item()
        return {"diag_mean": float(diag_mean), "offdiag_mean": float(offdiag_mean)}

    def forward(self, x):
        batch_size, time_steps, feat_dim = x.shape
        if feat_dim != self.feature_dim:
            raise ValueError("TemporalGraphBlock expects feature dim {}, got {}".format(self.feature_dim, feat_dim))

        h = x.reshape(batch_size, time_steps, self.num_channels, self.num_bands)

        adj_raw = 0.5 * (self.A_logits + self.A_logits.transpose(0, 1))
        if self.use_self_loop_prior:
            identity = torch.eye(self.num_channels, device=x.device, dtype=adj_raw.dtype)
            adj_raw = adj_raw + self.self_loop_weight * identity
        identity = torch.eye(self.num_channels, device=x.device, dtype=adj_raw.dtype)
        adj_stable = adj_raw * self.stable_adj_alpha + identity * (1.0 - self.stable_adj_alpha)
        a_norm = F.softmax(adj_stable, dim=-1)
        h_agg = torch.einsum("ij,btjf->btif", a_norm, h)
        if self.use_gcn_residual:
            h_msg = F.gelu(self.band_mlp(h_agg))
            h_graph = h + self.gcn_alpha * h_msg
        else:
            h_graph = self.band_mlp(h_agg)

        temporal_input = h_graph.reshape(batch_size, time_steps, self.feature_dim).transpose(1, 2)
        temporal_out = self.temporal_dw(temporal_input)
        temporal_out = self.temporal_pw(temporal_out)
        temporal_out = F.gelu(temporal_out)
        temporal_out = self.dropout(temporal_out)
        temporal_out = temporal_out.transpose(1, 2)

        out = self.norm(x + self.alpha * temporal_out)
        if self.use_pre_lstm_dropout:
            out = self.pre_lstm_dropout(out)
        return out
