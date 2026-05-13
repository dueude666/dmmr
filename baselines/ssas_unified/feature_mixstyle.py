import torch
import torch.nn as nn


class FeatureMixStyle(nn.Module):
    """Feature-level MixStyle for bottleneck vectors [B, D].

    The idea is borrowed from computer-vision domain generalization: mix feature
    statistics across samples so the classifier cannot rely on domain-specific style.
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps
        self.beta = torch.distributions.Beta(alpha, alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0 or torch.rand(1, device=x.device).item() > self.p:
            return x
        if x.dim() != 2 or x.size(0) < 2:
            return x

        mu = x.mean(dim=1, keepdim=True)
        sig = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt()
        x_norm = (x - mu) / sig
        perm = torch.randperm(x.size(0), device=x.device)
        lam = self.beta.sample((x.size(0), 1)).to(x.device)
        mu_mix = lam * mu + (1.0 - lam) * mu[perm]
        sig_mix = lam * sig + (1.0 - lam) * sig[perm]
        return x_norm * sig_mix + mu_mix
