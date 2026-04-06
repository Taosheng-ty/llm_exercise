"""
Exercise 06: Memory Budget Planner (Medium, numpy)

When planning LLM training, you need to know if your model fits in GPU memory.
The main memory consumers are:
- Model parameters
- Optimizer states (Adam stores m and v: 2x param memory in fp32)
- Gradients (same size as parameters)
- Activations (depends on batch_size, seq_len, model config)

Reference: slime memory_utils patterns

Your tasks:
-----------
1. Implement `param_memory_bytes(num_params, precision_bytes)`:
   - Return num_params * precision_bytes

2. Implement `optimizer_state_memory_bytes(num_params, optimizer_type, precision_bytes=4)`:
   - "adam" or "adamw": 2 states (m, v), each num_params * precision_bytes
   - "sgd": 0 extra (or 1 state if momentum, but assume 0 for simplicity)
   - "sgd_momentum": 1 state
   - Return total optimizer state memory.

3. Implement `gradient_memory_bytes(num_params, precision_bytes)`:
   - Same size as parameters.

4. Implement `activation_memory_bytes(batch_size, seq_len, hidden_dim, num_layers, num_heads, ffn_hidden_dim, precision_bytes=2)`:
   - Use simplified formula: per layer = batch_size * seq_len * (
       10 * hidden_dim + 2 * num_heads * seq_len + 4 * ffn_hidden_dim
     ) * precision_bytes
   - Total = num_layers * per_layer
   - (This is a rough estimate; exact depends on implementation)

5. Implement `total_training_memory_bytes(num_params, batch_size, seq_len, hidden_dim, num_layers, num_heads, ffn_hidden_dim, optimizer_type="adam", param_precision_bytes=2, optim_precision_bytes=4, activation_precision_bytes=2)`:
   - Sum of param + optimizer + gradient + activation memory.
   - Gradients are in param_precision_bytes.

6. Implement `max_batch_size(gpu_memory_bytes, num_params, seq_len, hidden_dim, num_layers, num_heads, ffn_hidden_dim, optimizer_type="adam", param_precision_bytes=2, optim_precision_bytes=4, activation_precision_bytes=2)`:
   - Find the maximum batch_size that fits in gpu_memory_bytes.
   - Binary search or linear scan from 1 upward.
   - Return 0 if even batch_size=1 doesn't fit.
"""

import numpy as np


def param_memory_bytes(num_params: int, precision_bytes: int) -> int:
    raise NotImplementedError


def optimizer_state_memory_bytes(
    num_params: int, optimizer_type: str, precision_bytes: int = 4
) -> int:
    raise NotImplementedError


def gradient_memory_bytes(num_params: int, precision_bytes: int) -> int:
    raise NotImplementedError


def activation_memory_bytes(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    ffn_hidden_dim: int,
    precision_bytes: int = 2,
) -> int:
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
