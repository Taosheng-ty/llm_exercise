"""
Exercise 07: KV Cache for Autoregressive Decoding
Difficulty: Medium

Implement a KV cache that stores past key and value tensors for efficient
autoregressive decoding.

Class KVCache:
    __init__(self, max_seq_len, num_heads, head_dim)
    update(self, new_k, new_v) -> (full_k, full_v)
        - new_k, new_v: (batch, num_heads, new_tokens, head_dim)
        - Appends to cache and returns full cached k, v
        - If cache exceeds max_seq_len, evict oldest tokens (rolling window)
    reset(self)
        - Clear the cache

Also implement:
    incremental_attention(Q, K, V, kv_cache)
        - Q: (batch, heads, 1, head_dim) - single new query
        - K, V: (batch, heads, 1, head_dim) - single new key/value
        - Updates cache and computes attention using full cached K, V
        - Returns output (batch, heads, 1, head_dim)
"""

import torch


class KVCache:
    def __init__(self, max_seq_len: int = 2048):
        # TODO: Initialize cache
        raise NotImplementedError

    def update(
        self, new_k: torch.Tensor, new_v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: Append to cache, handle overflow, return full k, v
        raise NotImplementedError

    def reset(self):
        # TODO: Clear cache
        raise NotImplementedError


def incremental_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    kv_cache: KVCache,
) -> torch.Tensor:
    # TODO: Update cache and compute attention
    raise NotImplementedError
