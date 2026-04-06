"""
Exercise 07: KV Cache - Solution
"""

import torch
import torch.nn.functional as F
import math


class KVCache:
    def __init__(self, max_seq_len: int = 2048):
        self.max_seq_len = max_seq_len
        self.k_cache = None  # Will be (batch, heads, cached_len, head_dim)
        self.v_cache = None

    def update(
        self, new_k: torch.Tensor, new_v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Append new_k, new_v to cache. Evict oldest if exceeding max_seq_len.

        new_k, new_v: (batch, heads, new_tokens, head_dim)
        Returns: (full_k, full_v) including all cached tokens
        """
        if self.k_cache is None:
            self.k_cache = new_k
            self.v_cache = new_v
        else:
            self.k_cache = torch.cat([self.k_cache, new_k], dim=2)
            self.v_cache = torch.cat([self.v_cache, new_v], dim=2)

        # Evict oldest if over max_seq_len
        if self.k_cache.size(2) > self.max_seq_len:
            self.k_cache = self.k_cache[:, :, -self.max_seq_len:, :]
            self.v_cache = self.v_cache[:, :, -self.max_seq_len:, :]

        return self.k_cache, self.v_cache

    def reset(self):
        self.k_cache = None
        self.v_cache = None

    @property
    def seq_len(self) -> int:
        if self.k_cache is None:
            return 0
        return self.k_cache.size(2)


def incremental_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    kv_cache: KVCache,
) -> torch.Tensor:
    """
    Single-step attention with KV cache.

    Q: (batch, heads, 1, head_dim) - new query
    K, V: (batch, heads, 1, head_dim) - new key/value
    """
    # Update cache and get full K, V
    full_k, full_v = kv_cache.update(K, V)

    # Compute attention: Q attends to all cached K
    d_k = Q.size(-1)
    scores = torch.matmul(Q, full_k.transpose(-2, -1)) / math.sqrt(d_k)
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, full_v)
    return out
