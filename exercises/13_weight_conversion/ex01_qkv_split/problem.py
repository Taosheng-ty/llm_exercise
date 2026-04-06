"""
Exercise 01: Split Fused QKV Weight into Separate Q, K, V (Medium, PyTorch)

In Megatron-LM, the query, key, and value projections are fused into a single
linear_qkv weight for efficiency. When converting to HuggingFace format, we need
to split this back into separate q_proj, k_proj, and v_proj weights.

The fused QKV weight is stored in an interleaved layout per query group:
  For each of the num_kv_heads groups, the weight contains:
    - (num_q_heads // num_kv_heads) rows of Q  (each row is head_dim wide in output)
    - 1 row of K
    - 1 row of V

Visual example (num_q_heads=4, num_kv_heads=2, head_dim=D, group_size=2):

  Fused QKV layout in memory (output dimension):
  +-------+-------+-------+-------+-------+-------+-------+-------+
  |  Q0   |  Q1   |  K0   |  V0   |  Q2   |  Q3   |  K1   |  V1   |
  | (D)   | (D)   | (D)   | (D)   | (D)   | (D)   | (D)   | (D)   |
  +-------+-------+-------+-------+-------+-------+-------+-------+
  |<--- KV group 0 ------------->|  |<--- KV group 1 ------------->|

  After split:
    Q = [Q0, Q1, Q2, Q3]  shape: (4*D, hidden_dim)
    K = [K0, K1]           shape: (2*D, hidden_dim)
    V = [V0, V1]           shape: (2*D, hidden_dim)

So the fused weight shape is:
  ((num_q_heads + 2 * num_kv_heads) * head_dim, hidden_dim)

After splitting:
  Q: (num_q_heads * head_dim, hidden_dim)
  K: (num_kv_heads * head_dim, hidden_dim)
  V: (num_kv_heads * head_dim, hidden_dim)

Reference: slime/backends/megatron_utils/megatron_to_hf/qwen3moe.py lines 66-76

Tasks:
    1. Implement split_qkv_weight() that splits fused QKV into separate Q, K, V tensors.
    2. Support GQA (Grouped Query Attention) where num_kv_heads <= num_q_heads.
    3. Implement fuse_qkv_weight() that reverses the split (for testing roundtrip).
"""

import torch


def split_qkv_weight(
    fused_qkv: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    hidden_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a fused QKV weight tensor into separate Q, K, V weight tensors.

    The fused weight is organized as num_kv_heads groups, each group containing:
      - (num_q_heads // num_kv_heads) Q head rows
      - 1 K head row
      - 1 V head row
    Each "row" is head_dim entries in the output dimension.

    Args:
        fused_qkv: shape ((num_q_heads + 2*num_kv_heads) * head_dim, hidden_dim)
        num_q_heads: number of query heads
        num_kv_heads: number of key/value heads (for GQA, num_kv_heads <= num_q_heads)
        head_dim: dimension per head
        hidden_dim: model hidden dimension

    Returns:
        (Q, K, V) where:
          Q: (num_q_heads * head_dim, hidden_dim)
          K: (num_kv_heads * head_dim, hidden_dim)
          V: (num_kv_heads * head_dim, hidden_dim)
    """
    # TODO: Implement this function
    # Hint: reshape to (num_kv_heads, group_size, head_dim, hidden_dim)
    # then use torch.split with appropriate sizes
    raise NotImplementedError


def fuse_qkv_weight(
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    hidden_dim: int,
) -> torch.Tensor:
    """Fuse separate Q, K, V weight tensors back into a single QKV tensor.

    This is the inverse of split_qkv_weight.

    Returns:
        fused_qkv: shape ((num_q_heads + 2*num_kv_heads) * head_dim, hidden_dim)
    """
    # TODO: Implement this function
    raise NotImplementedError
