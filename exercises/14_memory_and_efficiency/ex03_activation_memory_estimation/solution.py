"""
Solution for Exercise 03: Activation Memory Estimation
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
    """
    Estimate activation memory for multi-head attention in bytes.

    Stored activations:
    - Q, K, V after projection: each [batch, num_heads, seq_len, head_dim]
    - Attention scores (QK^T): [batch, num_heads, seq_len, seq_len]
    - Softmax output: [batch, num_heads, seq_len, seq_len]
    - Attention output (softmax * V): [batch, num_heads, seq_len, head_dim]
    """
    # Q, K, V: 3 tensors of shape [B, H, S, D]
    qkv_bytes = 3 * batch_size * num_heads * seq_len * head_dim * dtype_bytes

    # Attention scores + softmax: 2 tensors of shape [B, H, S, S]
    attn_scores_bytes = 2 * batch_size * num_heads * seq_len * seq_len * dtype_bytes

    # Attention output: [B, H, S, D]
    attn_output_bytes = batch_size * num_heads * seq_len * head_dim * dtype_bytes

    return qkv_bytes + attn_scores_bytes + attn_output_bytes


def estimate_ffn_activation_memory(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    ffn_hidden_dim: int,
    dtype_bytes: int = 2,
) -> int:
    """
    Estimate activation memory for FFN (SwiGLU-style) in bytes.

    SwiGLU FFN: out = W_down(SiLU(W_gate(x)) * W_up(x))
    Stored activations:
    - Gate projection output: [B, S, ffn_hidden_dim]
    - Up projection output: [B, S, ffn_hidden_dim]
    - Activation (SiLU) output: [B, S, ffn_hidden_dim]
    - Element-wise product (gate * up): [B, S, ffn_hidden_dim]
    """
    # gate, up, activation, product: 4 tensors of [B, S, ffn_hidden_dim]
    ffn_bytes = 4 * batch_size * seq_len * ffn_hidden_dim * dtype_bytes

    return ffn_bytes


def estimate_transformer_layer_memory(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_heads: int,
    ffn_hidden_dim: int,
    dtype_bytes: int = 2,
) -> int:
    """
    Estimate total activation memory for one transformer layer.
    Includes attention, FFN, and 2 residual connection saves.
    """
    head_dim = hidden_dim // num_heads

    attn_mem = estimate_attention_activation_memory(
        batch_size, num_heads, seq_len, head_dim, dtype_bytes
    )
    ffn_mem = estimate_ffn_activation_memory(
        batch_size, seq_len, hidden_dim, ffn_hidden_dim, dtype_bytes
    )

    # 2 residual saves: input to attention sublayer + input to FFN sublayer
    # Each: [B, S, hidden_dim]
    residual_mem = 2 * batch_size * seq_len * hidden_dim * dtype_bytes

    return attn_mem + ffn_mem + residual_mem


def estimate_total_activation_memory(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_heads: int,
    ffn_hidden_dim: int,
    num_layers: int,
    dtype_bytes: int = 2,
) -> int:
    """
    Estimate total activation memory for the full model.
    = num_layers * per_layer + embedding output.
    """
    per_layer = estimate_transformer_layer_memory(
        batch_size, seq_len, hidden_dim, num_heads, ffn_hidden_dim, dtype_bytes
    )

    # Embedding output: [B, S, hidden_dim]
    embedding_mem = batch_size * seq_len * hidden_dim * dtype_bytes

    return num_layers * per_layer + embedding_mem


def measure_actual_activation_memory(
    model: nn.Module, input_tensor: torch.Tensor
) -> int:
    """
    Use forward hooks to measure actual activation sizes during forward pass.
    Returns total bytes of all intermediate outputs.
    """
    total_bytes = [0]
    hooks = []

    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            total_bytes[0] += output.nelement() * output.element_size()
        elif isinstance(output, (tuple, list)):
            for o in output:
                if isinstance(o, torch.Tensor):
                    total_bytes[0] += o.nelement() * o.element_size()

    for module in model.modules():
        # Attach to leaf modules only to avoid double-counting
        if len(list(module.children())) == 0:
            h = module.register_forward_hook(hook_fn)
            hooks.append(h)

    with torch.no_grad():
        model(input_tensor)

    for h in hooks:
        h.remove()

    return total_bytes[0]
