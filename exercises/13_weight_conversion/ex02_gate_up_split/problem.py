"""
Exercise 02: Split Fused Gate-Up Projection (Easy, PyTorch)

Modern LLMs (LLaMA, Mistral, Qwen) use SwiGLU FFNs where the gate and up
projections are often fused into a single matrix for training efficiency.
Splitting them correctly is essential when converting checkpoints between
training frameworks and inference engines.

In Megatron-LM (and many MoE implementations), the gate and up projections of
the SwiGLU MLP are fused into a single linear_fc1 weight for efficiency:

  linear_fc1.weight shape: (2 * intermediate_size, hidden_size)

The first half is the gate projection and the second half is the up projection:
  gate_proj.weight: (intermediate_size, hidden_size)
  up_proj.weight:   (intermediate_size, hidden_size)

Reference: slime/backends/megatron_utils/megatron_to_hf/qwen3moe.py lines 31, 51, 93

Tasks:
    1. Implement split_gate_up() to split fused weight into gate and up projections.
    2. Implement fuse_gate_up() to merge them back.
    3. Implement split_moe_gate_up() to handle a batch of experts at once.
"""

import torch


def split_gate_up(
    fused_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a fused gate-up weight into separate gate_proj and up_proj.

    Args:
        fused_weight: shape (2 * intermediate_size, hidden_size)

    Returns:
        (gate_proj, up_proj) each of shape (intermediate_size, hidden_size)
    """
    # TODO: Implement this function
    raise NotImplementedError


def fuse_gate_up(
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
) -> torch.Tensor:
    """Fuse separate gate_proj and up_proj into a single weight.

    Args:
        gate_proj: shape (intermediate_size, hidden_size)
        up_proj: shape (intermediate_size, hidden_size)

    Returns:
        fused: shape (2 * intermediate_size, hidden_size)
    """
    # TODO: Implement this function
    raise NotImplementedError


def split_moe_gate_up(
    expert_weights: dict[int, torch.Tensor],
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Split fused gate-up weights for multiple MoE experts.

    Args:
        expert_weights: dict mapping expert_id -> fused_weight of shape
                        (2 * intermediate_size, hidden_size)

    Returns:
        dict mapping expert_id -> (gate_proj, up_proj)
    """
    # TODO: Implement this function
    raise NotImplementedError
