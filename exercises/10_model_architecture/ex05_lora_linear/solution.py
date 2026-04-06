"""
Solution for Exercise 05: LoRA Linear Layer
"""

import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int = 4, alpha: float = 1.0):
        super().__init__()
        self.linear = linear
        # Freeze original weights
        for param in self.linear.parameters():
            param.requires_grad = False

        in_features = linear.in_features
        out_features = linear.out_features

        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.scaling = alpha / r
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.linear(x)
        if not self.merged:
            # x: (..., in_features)
            # lora_A: (r, in_features) -> x @ A^T: (..., r)
            # lora_B: (out_features, r) -> (x @ A^T) @ B^T: (..., out_features)
            lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
            result = result + lora_out
        return result

    def merge(self):
        if not self.merged:
            self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def unmerge(self):
        if self.merged:
            self.linear.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
            self.merged = False
