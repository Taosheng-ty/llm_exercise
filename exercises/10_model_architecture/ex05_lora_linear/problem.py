"""
Exercise 05: LoRA (Low-Rank Adaptation) Linear Layer
=====================================================
Difficulty: Medium

LoRA adds trainable low-rank matrices to a frozen pretrained weight matrix.
Instead of fine-tuning W directly, we learn two small matrices A and B such that:

    output = W @ x + (B @ A) @ x * (alpha / r)

Where:
    - W is the frozen original weight (out_features, in_features)
    - A is (r, in_features), initialized with kaiming uniform
    - B is (out_features, r), initialized to zeros
    - alpha is a scaling hyperparameter
    - r is the rank

Your task:
    Implement LoRALinear that:
    1. Wraps an existing nn.Linear (frozen)
    2. Adds trainable lora_A and lora_B parameters
    3. Supports merge() and unmerge() for efficient inference
"""

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int = 4, alpha: float = 1.0):
        """
        Args:
            linear: pretrained nn.Linear to wrap (freeze its parameters)
            r: LoRA rank
            alpha: scaling factor
        """
        super().__init__()
        # TODO:
        # 1. Store the linear layer and freeze its parameters
        # 2. Create lora_A: Parameter of shape (r, in_features), kaiming uniform init
        # 3. Create lora_B: Parameter of shape (out_features, r), zero init
        # 4. Store scaling = alpha / r
        # 5. Track merged state
        raise NotImplementedError("Implement __init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        If not merged: output = linear(x) + (x @ A^T @ B^T) * scaling
        If merged: output = linear(x)  (LoRA already baked into weights)
        """
        # TODO
        raise NotImplementedError("Implement forward")

    def merge(self):
        """Merge LoRA weights into the frozen linear layer for inference."""
        # TODO: W += B @ A * scaling, set merged=True
        raise NotImplementedError("Implement merge")

    def unmerge(self):
        """Remove LoRA weights from the linear layer."""
        # TODO: W -= B @ A * scaling, set merged=False
        raise NotImplementedError("Implement unmerge")
