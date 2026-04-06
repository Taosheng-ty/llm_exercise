"""
Solution for Exercise 03: Exponential Moving Average (EMA) of Model Weights
"""

import torch
import torch.nn as nn
from copy import deepcopy


class EMAModel:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow_params = [p.clone().detach() for p in model.parameters()]

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_param, model_param in zip(self.shadow_params, model.parameters()):
            ema_param.mul_(self.decay).add_(model_param.data, alpha=1.0 - self.decay)

    def copy_to(self, model: nn.Module):
        for ema_param, model_param in zip(self.shadow_params, model.parameters()):
            model_param.data.copy_(ema_param.data)

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "shadow_params": [p.clone() for p in self.shadow_params],
        }

    def load_state_dict(self, state_dict: dict):
        self.decay = state_dict["decay"]
        self.shadow_params = [p.clone() for p in state_dict["shadow_params"]]
