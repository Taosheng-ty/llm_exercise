"""
Exercise 06: Merge LoRA Weights into Base Model (Medium, PyTorch)

LoRA (Low-Rank Adaptation) adds trainable low-rank matrices to frozen base weights:
  W_merged = W_base + (alpha / r) * B @ A

Where:
  - W_base: (out_features, in_features) original weight
  - A: (r, in_features) low-rank down projection
  - B: (out_features, r) low-rank up projection
  - alpha: scaling factor
  - r: rank

Tasks:
    1. Implement merge_lora() for a single weight.
    2. Implement unmerge_lora() to extract the delta (inverse operation).
    3. Implement merge_lora_state_dict() to merge all LoRA weights in a state dict.
       Convention: for base weight "model.layers.0.self_attn.q_proj.weight",
       LoRA keys are "model.layers.0.self_attn.q_proj.lora_A.weight" and
       "model.layers.0.self_attn.q_proj.lora_B.weight".
"""

import torch


def merge_lora(
    base_weight: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    alpha: float,
    r: int,
) -> torch.Tensor:
    """Merge LoRA weights into the base weight.

    Args:
        base_weight: (out_features, in_features)
        lora_A: (r, in_features)
        lora_B: (out_features, r)
        alpha: LoRA scaling factor
        r: LoRA rank

    Returns:
        merged weight: (out_features, in_features)
    """
    # TODO: Implement this function
    raise NotImplementedError


def unmerge_lora(
    merged_weight: torch.Tensor,
    base_weight: torch.Tensor,
    alpha: float,
    r: int,
) -> torch.Tensor:
    """Extract the LoRA delta from a merged weight.

    Returns:
        delta = merged_weight - base_weight, which equals (alpha/r) * B @ A
    """
    # TODO: Implement this function
    raise NotImplementedError


def merge_lora_state_dict(
    base_state_dict: dict[str, torch.Tensor],
    lora_state_dict: dict[str, torch.Tensor],
    alpha: float,
    r: int,
) -> dict[str, torch.Tensor]:
    """Merge all LoRA weights into a base state dict.

    For each base weight key like "model.layers.0.self_attn.q_proj.weight",
    look for corresponding LoRA keys:
      - "model.layers.0.self_attn.q_proj.lora_A.weight"
      - "model.layers.0.self_attn.q_proj.lora_B.weight"

    If both LoRA keys exist, merge them into the base weight.
    Base weights without LoRA counterparts are kept unchanged.

    Args:
        base_state_dict: the base model state dict
        lora_state_dict: the LoRA adapter state dict
        alpha: LoRA scaling factor
        r: LoRA rank

    Returns:
        merged state dict (same keys as base_state_dict)
    """
    # TODO: Implement this function
    raise NotImplementedError
