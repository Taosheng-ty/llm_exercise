"""
Solution for Exercise 06: Memory Budget Planner
"""

import numpy as np


def param_memory_bytes(num_params: int, precision_bytes: int) -> int:
    return num_params * precision_bytes


def optimizer_state_memory_bytes(
    num_params: int, optimizer_type: str, precision_bytes: int = 4
) -> int:
    opt = optimizer_type.lower()
    if opt in ("adam", "adamw"):
        # m and v states, each num_params * precision_bytes
        return 2 * num_params * precision_bytes
    elif opt == "sgd_momentum":
        return num_params * precision_bytes
    elif opt == "sgd":
        return 0
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def gradient_memory_bytes(num_params: int, precision_bytes: int) -> int:
    return num_params * precision_bytes


def activation_memory_bytes(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    ffn_hidden_dim: int,
    precision_bytes: int = 2,
) -> int:
    """
    Simplified activation memory estimate per layer:
    batch_size * seq_len * (10*hidden_dim + 2*num_heads*seq_len + 4*ffn_hidden_dim) * precision_bytes
    """
    per_layer = batch_size * seq_len * (
        10 * hidden_dim + 2 * num_heads * seq_len + 4 * ffn_hidden_dim
    ) * precision_bytes
    return num_layers * per_layer


def total_training_memory_bytes(
    num_params: int,
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    ffn_hidden_dim: int,
    optimizer_type: str = "adam",
    param_precision_bytes: int = 2,
    optim_precision_bytes: int = 4,
    activation_precision_bytes: int = 2,
) -> int:
    p_mem = param_memory_bytes(num_params, param_precision_bytes)
    o_mem = optimizer_state_memory_bytes(num_params, optimizer_type, optim_precision_bytes)
    g_mem = gradient_memory_bytes(num_params, param_precision_bytes)
    a_mem = activation_memory_bytes(
        batch_size, seq_len, hidden_dim, num_layers, num_heads,
        ffn_hidden_dim, activation_precision_bytes
    )
    return p_mem + o_mem + g_mem + a_mem


def max_batch_size(
    gpu_memory_bytes: int,
    num_params: int,
    seq_len: int,
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    ffn_hidden_dim: int,
    optimizer_type: str = "adam",
    param_precision_bytes: int = 2,
    optim_precision_bytes: int = 4,
    activation_precision_bytes: int = 2,
) -> int:
    # Fixed costs (don't depend on batch_size)
    fixed = (
        param_memory_bytes(num_params, param_precision_bytes)
        + optimizer_state_memory_bytes(num_params, optimizer_type, optim_precision_bytes)
        + gradient_memory_bytes(num_params, param_precision_bytes)
    )

    remaining = gpu_memory_bytes - fixed
    if remaining <= 0:
        return 0

    # Activation memory per sample (batch_size=1)
    per_sample = activation_memory_bytes(
        1, seq_len, hidden_dim, num_layers, num_heads,
        ffn_hidden_dim, activation_precision_bytes
    )

    if per_sample <= 0:
        return 0

    max_bs = remaining // per_sample
    return max(0, max_bs)
