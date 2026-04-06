"""
Solution for Exercise 03: ALiBi Attention
"""

import math

import torch
import torch.nn.functional as F


def compute_alibi_slopes(num_heads: int) -> torch.Tensor:
    """
    Compute ALiBi slopes as a geometric sequence.

    slopes[i] = 2^(-8 * (i+1) / num_heads)

    Returns:
        Tensor of shape (num_heads,)
    """
    exponents = torch.arange(1, num_heads + 1, dtype=torch.float32)
    slopes = 2.0 ** (-8.0 * exponents / num_heads)
    return slopes


def compute_alibi_bias(seq_len: int, num_heads: int) -> torch.Tensor:
    """
    Compute the ALiBi distance bias matrix.

    bias[h, q, k] = slopes[h] * (k - q)

    Returns:
        Tensor of shape (num_heads, seq_len, seq_len)
    """
    slopes = compute_alibi_slopes(num_heads)  # (num_heads,)

    # Distance matrix: (k - q) for each (q, k) pair
    q_pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)  # (seq_len, 1)
    k_pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0)  # (1, seq_len)
    distance = k_pos - q_pos  # (seq_len, seq_len), negative for k < q

    # bias[h, q, k] = slopes[h] * distance[q, k]
    # slopes: (H,) -> (H, 1, 1)
    bias = slopes.view(-1, 1, 1) * distance.unsqueeze(0)  # (H, S, S)

    return bias


def alibi_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    alibi_bias: torch.Tensor,
    causal_mask: bool = True,
) -> torch.Tensor:
    """
    Scaled dot-product attention with ALiBi positional bias.

    Args:
        Q: (batch, num_heads, seq_len, head_dim)
        K: (batch, num_heads, seq_len, head_dim)
        V: (batch, num_heads, seq_len, head_dim)
        alibi_bias: (num_heads, seq_len, seq_len) or broadcastable
        causal_mask: whether to apply causal masking

    Returns:
        Output of shape (batch, num_heads, seq_len, head_dim)
    """
    head_dim = Q.size(-1)
    seq_len = Q.size(-2)

    # Scaled dot-product
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)

    # Add ALiBi bias (broadcasts over batch)
    scores = scores + alibi_bias

    # Apply causal mask
    if causal_mask:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    # Softmax and weighted sum
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)

    return output
