"""
Exercise 03: Complete Transformer Decoder Block
================================================
Difficulty: Hard

A modern transformer decoder block (as in LLaMA, GPT, etc.) uses:

    Pre-norm architecture:
        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))

    Where Attention is multi-head self-attention with causal masking, and
    FFN is a SwiGLU feed-forward network.

Your task:
    Implement ALL of the following from scratch:
    1. RMSNorm
    2. Multi-head causal self-attention (with Q, K, V projections and output projection)
    3. SwiGLU FFN
    4. TransformerBlock that combines them with pre-norm residual connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        # TODO
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO
        raise NotImplementedError


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        """
        Args:
            dim: model dimension
            n_heads: number of attention heads (dim must be divisible by n_heads)
        """
        super().__init__()
        # TODO:
        # 1. Store n_heads, head_dim = dim // n_heads
        # 2. Create Q, K, V, and output projections (nn.Linear, no bias)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
        Returns:
            (batch, seq_len, dim)
        """
        # TODO:
        # 1. Project to Q, K, V
        # 2. Reshape to (batch, n_heads, seq_len, head_dim)
        # 3. Compute scaled dot-product attention with causal mask
        # 4. Concatenate heads and project output
        raise NotImplementedError


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        # TODO
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO
        raise NotImplementedError


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ffn_hidden_dim: int):
        """
        Args:
            dim: model dimension
            n_heads: number of attention heads
            ffn_hidden_dim: hidden dimension for FFN
        """
        super().__init__()
        # TODO:
        # 1. attention_norm (RMSNorm)
        # 2. attention (CausalSelfAttention)
        # 3. ffn_norm (RMSNorm)
        # 4. ffn (SwiGLUFFN)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pre-norm transformer block:
            x = x + attention(attention_norm(x))
            x = x + ffn(ffn_norm(x))
        """
        # TODO
        raise NotImplementedError
