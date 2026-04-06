"""
Exercise 01: Scaled Dot-Product Attention - Solution
"""

import torch
import torch.nn.functional as F
import math


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Implement scaled dot-product attention.

    Args:
        Q: (batch, heads, seq_len, head_dim)
        K: (batch, heads, seq_len, head_dim)
        V: (batch, heads, seq_len, head_dim)
        causal: whether to apply causal mask

    Returns:
        output: (batch, heads, seq_len, head_dim)
        attention_weights: (batch, heads, seq_len, seq_len)
    """
    d_k = Q.size(-1)
    # score = Q @ K^T / sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if causal:
        seq_len = Q.size(-2)
        # Create lower-triangular causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output, attention_weights
