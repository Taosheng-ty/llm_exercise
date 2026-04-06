"""
Solution for Exercise 06: Merge LoRA Weights into Base Model
"""

import torch


def merge_lora(
    base_weight: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    alpha: float,
    r: int,
) -> torch.Tensor:
    """Merge LoRA weights into the base weight."""
    scaling = alpha / r
    return base_weight + scaling * (lora_B @ lora_A)


def unmerge_lora(
    merged_weight: torch.Tensor,
    base_weight: torch.Tensor,
    alpha: float,
    r: int,
) -> torch.Tensor:
    """Extract the LoRA delta from a merged weight."""
    return merged_weight - base_weight


def merge_lora_state_dict(
    base_state_dict: dict[str, torch.Tensor],
    lora_state_dict: dict[str, torch.Tensor],
    alpha: float,
    r: int,
) -> dict[str, torch.Tensor]:
    """Merge all LoRA weights into a base state dict."""
    merged = {}

    for key, base_weight in base_state_dict.items():
        # Check if this weight has LoRA adapters
        # Strip ".weight" suffix, add ".lora_A.weight" / ".lora_B.weight"
        if key.endswith(".weight"):
            prefix = key[: -len(".weight")]
            lora_a_key = f"{prefix}.lora_A.weight"
            lora_b_key = f"{prefix}.lora_B.weight"

            if lora_a_key in lora_state_dict and lora_b_key in lora_state_dict:
                lora_A = lora_state_dict[lora_a_key]
                lora_B = lora_state_dict[lora_b_key]
                merged[key] = merge_lora(base_weight, lora_A, lora_B, alpha, r)
                continue

        merged[key] = base_weight.clone()

    return merged
