"""
Solution for Exercise 02: Split Fused Gate-Up Projection
"""

import torch


def split_gate_up(
    fused_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a fused gate-up weight into separate gate_proj and up_proj."""
    gate_proj, up_proj = fused_weight.chunk(2, dim=0)
    return gate_proj, up_proj


def fuse_gate_up(
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
) -> torch.Tensor:
    """Fuse separate gate_proj and up_proj into a single weight."""
    return torch.cat([gate_proj, up_proj], dim=0)


def split_moe_gate_up(
    expert_weights: dict[int, torch.Tensor],
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Split fused gate-up weights for multiple MoE experts."""
    result = {}
    for expert_id, fused_weight in expert_weights.items():
        result[expert_id] = split_gate_up(fused_weight)
    return result
