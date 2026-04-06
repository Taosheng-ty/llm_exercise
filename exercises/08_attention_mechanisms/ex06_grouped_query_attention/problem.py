"""
Exercise 06: Grouped Query Attention (GQA)
Difficulty: Medium

Implement GQA where K,V have fewer heads than Q. GQA (used in LLaMA 2/3,
Mistral) reduces the number of KV heads while keeping full query heads,
dramatically cutting KV cache memory during inference without sacrificing
quality. This is essential for serving LLMs efficiently, especially during RL
rollout generation where many sequences are decoded in parallel.

K,V heads are shared across groups of query heads.
- num_q_heads must be divisible by num_kv_heads
- Each KV head serves (num_q_heads // num_kv_heads) query heads

Implement:
  grouped_query_attention(Q, K, V, num_q_heads, num_kv_heads, causal=False)

Args:
    Q: (batch, seq_len, num_q_heads * head_dim)
    K: (batch, seq_len, num_kv_heads * head_dim)
    V: (batch, seq_len, num_kv_heads * head_dim)
    num_q_heads: number of query heads
    num_kv_heads: number of key/value heads
    causal: whether to apply causal mask

Returns:
    output: (batch, seq_len, num_q_heads * head_dim)
"""

import torch


def grouped_query_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    # TODO: Implement grouped query attention
    raise NotImplementedError
