"""
Exercise 07: Residual Connections with Scaling
===============================================
Difficulty: Easy

Residual connections are fundamental to training deep transformers. The pre-norm
residual pattern is:

    output = x + sublayer(norm(x))

Some deep networks (e.g., DeepNorm) use residual scaling:

    output = x * alpha + sublayer(norm(x))

Your task:
    Implement PreNormResidual that wraps any sublayer with:
    1. Layer normalization (RMSNorm)
    2. Residual connection
    3. Optional scaling factor alpha
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMSNorm -- you may copy from ex01 or reimplement.

    RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight

    Create self.weight as nn.Parameter of ones(dim), store self.eps.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """Store eps, create self.weight = nn.Parameter(torch.ones(dim))."""
        super().__init__()
        # TODO
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize x using RMS and scale by self.weight."""
        # TODO
        raise NotImplementedError


class PreNormResidual(nn.Module):
    def __init__(self, dim: int, sublayer: nn.Module, alpha: float = 1.0):
        """
        Args:
            dim: dimension for RMSNorm
            sublayer: any nn.Module to wrap
            alpha: residual scaling factor (default 1.0 = no scaling)
        """
        super().__init__()
        # TODO: store norm, sublayer, alpha
        raise NotImplementedError("Implement __init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor
        Returns:
            x * alpha + sublayer(norm(x))
        """
        # TODO
        raise NotImplementedError("Implement forward")
