"""
Exercise 01: RMSNorm (Root Mean Square Layer Normalization)
===========================================================
Difficulty: Easy

RMSNorm is a simpler alternative to LayerNorm used in modern LLMs (LLaMA, Qwen, etc.).
Unlike LayerNorm, it does NOT subtract the mean -- it only normalizes by the root mean
square of the input, then scales by a learnable weight vector.

Formula:
    rms(x) = sqrt(mean(x^2) + eps)
    output = (x / rms(x)) * weight

Your task:
    Implement the RMSNorm class below. It should:
    1. Accept normalized_shape (int) and eps (float) in __init__
    2. Have a learnable `weight` parameter of shape (normalized_shape,), initialized to ones
    3. Forward: normalize x by its RMS across the last dimension, then multiply by weight
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        # TODO: create a learnable weight parameter of shape (normalized_shape,)
        # initialized to ones
        raise NotImplementedError("Implement __init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (..., normalized_shape)
        Returns:
            Normalized tensor of the same shape
        """
        # TODO: compute RMS norm
        # 1. Compute mean of x^2 along last dimension (keep dims)
        # 2. Add eps, take sqrt -> this is the RMS
        # 3. Divide x by RMS
        # 4. Multiply by self.weight
        raise NotImplementedError("Implement forward")
