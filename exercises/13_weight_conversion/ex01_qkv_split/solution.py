"""
Solution for Exercise 01: Split Fused QKV Weight into Separate Q, K, V
"""

import torch


def split_qkv_weight(
    fused_qkv: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    hidden_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a fused QKV weight tensor into separate Q, K, V weight tensors."""
    # Number of Q heads per KV group
    q_per_group = num_q_heads // num_kv_heads

    # Reshape: (num_kv_heads, q_per_group + 1 + 1, head_dim, hidden_dim)
    param = fused_qkv.view(num_kv_heads, q_per_group + 2, head_dim, hidden_dim)

    # Split along dim=1: Q gets q_per_group, K gets 1, V gets 1
    q_param, k_param, v_param = torch.split(
        param, split_size_or_sections=[q_per_group, 1, 1], dim=1
    )

    # Reshape back to 2D
    q_param = q_param.reshape(num_q_heads * head_dim, hidden_dim)
    k_param = k_param.reshape(num_kv_heads * head_dim, hidden_dim)
    v_param = v_param.reshape(num_kv_heads * head_dim, hidden_dim)

    return q_param, k_param, v_param


def fuse_qkv_weight(
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    hidden_dim: int,
) -> torch.Tensor:
    """Fuse separate Q, K, V weight tensors back into a single QKV tensor."""
    q_per_group = num_q_heads // num_kv_heads

    # Reshape each to group form
    q = q_weight.view(num_kv_heads, q_per_group, head_dim, hidden_dim)
    k = k_weight.view(num_kv_heads, 1, head_dim, hidden_dim)
    v = v_weight.view(num_kv_heads, 1, head_dim, hidden_dim)

    # Concatenate along group dimension
    fused = torch.cat([q, k, v], dim=1)

    # Flatten back to 2D
    total_rows = (num_q_heads + 2 * num_kv_heads) * head_dim
    return fused.reshape(total_rows, hidden_dim)
