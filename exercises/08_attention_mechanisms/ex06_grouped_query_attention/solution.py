"""
Exercise 06: Grouped Query Attention (GQA) - Solution
"""

import torch
import torch.nn.functional as F
import math


def grouped_query_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    """
    Grouped Query Attention: K,V heads are shared across query head groups.

    Q: (batch, seq_len, num_q_heads * head_dim)
    K: (batch, seq_len, num_kv_heads * head_dim)
    V: (batch, seq_len, num_kv_heads * head_dim)
    """
    assert num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads"
    B, S, _ = Q.shape
    head_dim = Q.size(-1) // num_q_heads
    num_groups = num_q_heads // num_kv_heads

    # Reshape Q: (B, num_q_heads, S, head_dim)
    Q = Q.view(B, S, num_q_heads, head_dim).transpose(1, 2)
    # Reshape K, V: (B, num_kv_heads, S, head_dim)
    K = K.view(B, S, num_kv_heads, head_dim).transpose(1, 2)
    V = V.view(B, S, num_kv_heads, head_dim).transpose(1, 2)

    # Expand K, V to match Q heads by repeating each KV head `num_groups` times
    # (B, num_kv_heads, S, head_dim) -> (B, num_q_heads, S, head_dim)
    K = K.unsqueeze(2).expand(B, num_kv_heads, num_groups, S, head_dim)
    K = K.reshape(B, num_q_heads, S, head_dim)
    V = V.unsqueeze(2).expand(B, num_kv_heads, num_groups, S, head_dim)
    V = V.reshape(B, num_q_heads, S, head_dim)

    # Standard scaled dot-product attention
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)

    if causal:
        mask = torch.triu(torch.ones(S, S, device=Q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, V)  # (B, num_q_heads, S, head_dim)

    # Reshape back: (B, S, num_q_heads * head_dim)
    out = out.transpose(1, 2).contiguous().view(B, S, num_q_heads * head_dim)
    return out
