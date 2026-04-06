"""
Exercise 05: FLOPs Counter (Medium)

Counting floating point operations (FLOPs) is essential for:
- Estimating training time
- Computing hardware utilization (MFU - Model FLOPS Utilization)
- Comparing model architectures

Standard conventions:
- A matrix multiply of [M, K] x [K, N] = 2*M*K*N FLOPs (multiply + add)
- Attention QK^T with causal mask: 2*H*S*S*D / 2 (half due to causal mask)
- A*V: H*S*S*D (no factor of 2 for single matmul with causal)
- Training FLOPs ~= 3x forward FLOPs (forward + backward, where backward ~ 2x forward)

Reference: slime/utils/flops_utils.py

Your tasks:
-----------
1. Implement `linear_flops(batch_size, seq_len, in_features, out_features)`:
   - FLOPs for a linear layer: 2 * batch_size * seq_len * in_features * out_features

2. Implement `attention_flops(batch_size, num_heads, seq_len, head_dim)`:
   - QK^T (causal): 2 * num_heads * seq_len * seq_len * head_dim / 2
   - softmax*V: num_heads * seq_len * seq_len * head_dim
   - Multiply by batch_size.
   - Return total as int.

3. Implement `ffn_flops(batch_size, seq_len, hidden_dim, ffn_hidden_dim)`:
   - Standard FFN with gate (SwiGLU): 3 projections (gate, up, down)
   - 2 * batch_size * seq_len * hidden_dim * ffn_hidden_dim * 3

4. Implement `transformer_block_flops(batch_size, seq_len, hidden_dim, num_heads, ffn_hidden_dim)`:
   - QKV projections: 3 linear layers (hidden_dim -> hidden_dim)
   - Attention: attention_flops
   - Output projection: linear (hidden_dim -> hidden_dim)
   - FFN: ffn_flops

5. Implement `total_training_flops(batch_size, seq_len, hidden_dim, num_heads, ffn_hidden_dim, num_layers, vocab_size)`:
   - Sum of all layer flops + LM head (linear: hidden_dim -> vocab_size)
   - Multiply by 3 for training (forward + backward)
"""

import torch


def linear_flops(
    batch_size: int, seq_len: int, in_features: int, out_features: int
) -> int:
    """FLOPs for a linear layer."""
    raise NotImplementedError


def attention_flops(
    batch_size: int, num_heads: int, seq_len: int, head_dim: int
) -> int:
    """FLOPs for multi-head attention (causal)."""
    raise NotImplementedError


def ffn_flops(
    batch_size: int, seq_len: int, hidden_dim: int, ffn_hidden_dim: int
) -> int:
    """FLOPs for SwiGLU FFN (3 projections)."""
    raise NotImplementedError


def transformer_block_flops(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_heads: int,
    ffn_hidden_dim: int,
) -> int:
    """FLOPs for one transformer block (attention + FFN)."""
    raise NotImplementedError


def total_training_flops(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_heads: int,
    ffn_hidden_dim: int,
    num_layers: int,
    vocab_size: int,
) -> int:
    """Total training FLOPs (3x forward for training)."""
    raise NotImplementedError
