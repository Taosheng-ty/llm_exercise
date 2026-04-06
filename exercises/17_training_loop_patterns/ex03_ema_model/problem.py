"""
Exercise 03: Exponential Moving Average (EMA) of Model Weights

Difficulty: Medium
Framework: PyTorch

Background:
    EMA maintains a smoothed copy of model weights that often generalizes
    better than the raw trained weights. After each training step, EMA
    parameters are updated:

        ema_param = decay * ema_param + (1 - decay) * model_param

    Typical decay values are 0.999 or 0.9999.

    The EMA model is used at evaluation time by copying EMA weights into
    the model (copy_to), evaluating, then restoring original weights.

Implement the EMAModel class with these methods:
    __init__(model, decay): Initialize EMA params as copies of model params
    update(model): Perform one EMA update step
    copy_to(model): Copy EMA params into model (for evaluation)
    state_dict() -> dict: Return EMA parameters as a state dict
    load_state_dict(state_dict): Load EMA parameters from a state dict
"""

import torch
import torch.nn as nn
from copy import deepcopy


class EMAModel:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        """
        Args:
            model: the model whose parameters to track
            decay: EMA decay rate (higher = slower update)
        """
        self.decay = decay
        # TODO: Store a deep copy of model parameters as EMA shadow params
        # Hint: Use list(model.parameters()) and .clone().detach() each one
        raise NotImplementedError("Implement EMAModel.__init__")

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA parameters with current model parameters."""
        # TODO: For each (ema_param, model_param) pair:
        #   ema_param = decay * ema_param + (1 - decay) * model_param
        # Hint: Use .mul_() and .add_() for in-place operations
        raise NotImplementedError("Implement EMAModel.update")

    def copy_to(self, model: nn.Module):
        """Copy EMA parameters into model (for evaluation)."""
        # TODO: Copy each ema_param's data into the corresponding model param
        raise NotImplementedError("Implement EMAModel.copy_to")

    def state_dict(self) -> dict:
        """Return EMA state for checkpointing."""
        # TODO: Return dict with 'decay' and 'shadow_params'
        raise NotImplementedError("Implement EMAModel.state_dict")

    def load_state_dict(self, state_dict: dict):
        """Load EMA state from checkpoint."""
        # TODO: Restore decay and shadow_params
        raise NotImplementedError("Implement EMAModel.load_state_dict")
