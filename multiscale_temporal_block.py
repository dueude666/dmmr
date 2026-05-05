import torch
import torch.nn as nn


class MultiScaleSpatiotemporalBlock(nn.Module):
    def __init__(self, input_dim=310, alpha_init=0.1):
        super(MultiScaleSpatiotemporalBlock, self).__init__()
        self.input_dim = int(input_dim)
        self.branch_k3 = nn.Conv1d(self.input_dim, self.input_dim, kernel_size=3, padding=1)
        self.branch_k5 = nn.Conv1d(self.input_dim, self.input_dim, kernel_size=5, padding=2)
        self.branch_k7 = nn.Conv1d(self.input_dim, self.input_dim, kernel_size=7, padding=3)
        self.fuse = nn.Conv1d(self.input_dim * 3, self.input_dim, kernel_size=1)
        self.norm = nn.LayerNorm(self.input_dim)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))

    def get_alpha_value(self):
        return float(self.alpha.detach().cpu().item())

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("MST expects [B, T, C], got {}".format(tuple(x.shape)))
        if x.size(-1) != self.input_dim:
            raise ValueError("MST expects input_dim {}, got {}".format(self.input_dim, x.size(-1)))

        x_t = x.transpose(1, 2)
        b3 = self.branch_k3(x_t)
        b5 = self.branch_k5(x_t)
        b7 = self.branch_k7(x_t)
        fused = self.fuse(torch.cat([b3, b5, b7], dim=1))
        fused = fused.transpose(1, 2)
        out = self.norm(x + self.alpha * fused)
        return out
