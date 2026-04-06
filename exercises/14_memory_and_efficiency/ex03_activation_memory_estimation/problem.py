"""
Exercise 03: Activation Memory Estimation (Medium)

During training, we need to store intermediate activations for the backward pass.
Knowing how much memory activations consume helps with:
- Choosing batch sizes
- Deciding when to use gradient checkpointing
- Planning GPU memory budgets

For a standard transformer layer, key activations include:
- Input to each sub-layer (for residual connections)
- Attention scores: [batch, num_heads, seq_len, seq_len]
- Attention output (after softmax * V): [batch, num_heads, seq_len, head_dim]
- FFN intermediate: [batch, seq_len, ffn_hidden_dim]
- Layer norm inputs

Your tasks:
-----------
1. Implement `estimate_attention_activation_memory(batch_size, num_heads, seq_len, head_dim, dtype_bytes=2)`:
   - Compute bytes for: Q, K, V projections, attention scores (QK^T), softmax output, attention output (softmax * V)
   - Return total bytes as an integer.

2. Implement `estimate_ffn_activation_memory(batch_size, seq_len, hidden_dim, ffn_hidden_dim, dtype_bytes=2)`:
   - Compute bytes for: input projection, gate projection (for SwiGLU-style), activation output, down projection input
   - Return total bytes.

3. Implement `estimate_transformer_layer_memory(batch_size, seq_len, hidden_dim, num_heads, ffn_hidden_dim, dtype_bytes=2)`:
   - Sum attention + FFN + residual activations (2 residual saves per layer).
   - Return total bytes.

4. Implement `estimate_total_activation_memory(batch_size, seq_len, hidden_dim, num_heads, ffn_hidden_dim, num_layers, dtype_bytes=2)`:
   - Total for all layers plus embedding output.
   - Return total bytes.

5. Implement `measure_actual_activation_memory(model, input_tensor)`:
   - Use forward hooks to measure actual tensor sizes stored during forward pass.
   - Hook non-container (leaf) submodules only (modules with no children).
   - Count the OUTPUT tensors of each hooked module (not inputs).
   - Return total bytes measured.
"""

import torch
import torch.nn as nn


def estimate_attention_activation_memory(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    dtype_bytes: int = 2,
) -> int:
    """Estimate activation memory for multi-head attention in bytes."""
    raise NotImplementedError


def estimate_ffn_activation_memory(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    ffn_hidden_dim: int,
    dtype_bytes: int = 2,
) -> int:
    """Estimate activation memory for FFN (SwiGLU-style) in bytes."""
    raise NotImplementedError


def estimate_transformer_layer_memory(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_heads: int,
    ffn_hidden_dim: int,
    dtype_bytes: int = 2,
) -> int:
    """Estimate total activation memory for one transformer layer in bytes."""
    raise NotImplementedError


def estimate_total_activation_memory(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_heads: int,
    ffn_hidden_dim: int,
    num_layers: int,
    dtype_bytes: int = 2,
) -> int:
    """Estimate total activation memory for the full model in bytes."""
    raise NotImplementedError


def measure_actual_activation_memory(
    model: nn.Module, input_tensor: torch.Tensor
) -> int:
    """
    Use forward hooks to measure actual activation sizes during forward pass.
    Returns total bytes of all intermediate outputs.
    """
    raise NotImplementedError
