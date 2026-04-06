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
    """Root Mean Square Layer Normalization.

    RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight

    where weight is a learnable parameter initialized to ones.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """Create self.weight as nn.Parameter of ones(dim), store eps."""
        super().__init__()
        # TODO
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize x using RMS and scale by self.weight."""
        # TODO
        raise NotImplementedError


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with causal masking.

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(head_dim) + causal_mask) @ V

    The causal mask sets future positions to -inf so each token can only
    attend to itself and previous tokens.
    """

    def __init__(self, dim: int, n_heads: int):
        """
        Args:
            dim: model dimension
            n_heads: number of attention heads (dim must be divisible by n_heads)

        Create:
            self.n_heads, self.head_dim = n_heads, dim // n_heads
            self.q_proj, self.k_proj, self.v_proj, self.o_proj: nn.Linear(dim, dim, bias=False)
        """
        super().__init__()
        # TODO:
        # 1. Store n_heads, head_dim = dim // n_heads
        # 2. Create projections (nn.Linear, no bias) named:
        #    self.q_proj, self.k_proj, self.v_proj, self.o_proj
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
        Returns:
            (batch, seq_len, dim)

        Steps:
            1. Project to Q, K, V: each (batch, seq_len, dim)
            2. Reshape to (batch, n_heads, seq_len, head_dim)
            3. Compute attention scores: Q @ K^T / sqrt(head_dim)
            4. Apply causal mask: set upper triangle to -inf
            5. Softmax over last dim, then @ V
            6. Reshape back to (batch, seq_len, dim) and apply o_proj
        """
        # TODO
        raise NotImplementedError


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network.

    output = down_proj(silu(gate_proj(x)) * up_proj(x))

    Create three nn.Linear layers (no bias):
        self.gate_proj: dim -> hidden_dim
        self.up_proj:   dim -> hidden_dim
        self.down_proj: hidden_dim -> dim
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        # TODO
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return down_proj(silu(gate_proj(x)) * up_proj(x))."""
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
