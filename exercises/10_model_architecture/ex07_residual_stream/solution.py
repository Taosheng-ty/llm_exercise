"""
Solution for Exercise 07: Residual Connections with Scaling
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * rms).to(dtype) * self.weight


class PreNormResidual(nn.Module):
    def __init__(self, dim: int, sublayer: nn.Module, alpha: float = 1.0):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.sublayer = sublayer
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.alpha + self.sublayer(self.norm(x))
