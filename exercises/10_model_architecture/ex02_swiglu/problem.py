"""
Exercise 02: SwiGLU Activation
===============================
Difficulty: Easy

SwiGLU is a gated activation function used in modern LLMs (LLaMA, Qwen, Mistral, etc.)
as the feed-forward network (FFN) activation. It replaces the traditional ReLU MLP.

Formula:
    SwiGLU(x) = (x @ W1) * silu(x @ W_gate)
    where silu(x) = x * sigmoid(x)

The FFN typically has:
    - W_gate: (dim, hidden_dim)  -- gate projection
    - W1:     (dim, hidden_dim)  -- up projection
    - W2:     (hidden_dim, dim)  -- down projection

    output = (silu(x @ W_gate) * (x @ W1)) @ W2

Your task:
    Implement the SwiGLUFFN class. It should:
    1. Have three linear layers: gate_proj, up_proj, down_proj (no bias)
    2. Forward: apply silu gating as described above
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        """
        Args:
            dim: input/output dimension
            hidden_dim: intermediate hidden dimension
        """
        super().__init__()
        # TODO: create three nn.Linear layers (no bias):
        #   gate_proj: dim -> hidden_dim
        #   up_proj:   dim -> hidden_dim
        #   down_proj: hidden_dim -> dim
        raise NotImplementedError("Implement __init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
        Returns:
            (batch, seq_len, dim)
        """
        # TODO:
        # 1. gate = silu(gate_proj(x))    -- use F.silu
        # 2. up = up_proj(x)
        # 3. return down_proj(gate * up)
        raise NotImplementedError("Implement forward")
