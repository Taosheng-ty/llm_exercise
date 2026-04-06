"""
Exercise 02: Mixed Precision Training (Medium)

Mixed precision training uses lower precision (fp16/bf16) for forward/backward passes
while keeping a master copy of weights in fp32 for the optimizer step. This reduces
memory usage and speeds up computation on modern GPUs.

Key components:
- Forward pass in fp16 (or bf16)
- Loss scaling to prevent underflow of small gradients in fp16
- Gradient unscaling before optimizer step
- Inf/NaN detection to skip bad steps and adjust scale factor
- fp32 master weights for numerically stable optimizer updates

Your tasks:
-----------
1. Implement `GradScaler`:
   - __init__(self, init_scale=2**16, growth_factor=2.0, backoff_factor=0.5,
              growth_interval=2000):
       Store scale, factors, and a counter for consecutive good steps.

   - scale(self, loss) -> scaled_loss:
       Multiply loss by current scale factor. Return scaled loss.

   - unscale_(self, optimizer):
       Divide all gradients in optimizer.param_groups by current scale.
       Set self._found_inf to True if any gradient contains inf or nan.

   - step(self, optimizer):
       If no inf/nan found, call optimizer.step().
       Otherwise skip the step.

   - update(self):
       If inf/nan was found: scale *= backoff_factor, reset counter.
       Else: increment counter; if counter >= growth_interval: scale *= growth_factor, reset counter.

2. Implement `mixed_precision_train_step(model, optimizer, data, target, loss_fn, scaler)`:
   - Cast model inputs to float16 for the forward pass.
   - Compute loss in fp16.
   - Use scaler to scale loss, backward, unscale, step, and update.
   - Return the (unscaled) loss value as a float.
"""

import torch
import torch.nn as nn


class GradScaler:
    """
    Simplified gradient scaler for mixed precision training.
    """

    def __init__(
        self,
        init_scale: float = 2**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ):
        raise NotImplementedError("Implement GradScaler.__init__")

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Multiply loss by current scale factor."""
        raise NotImplementedError("Implement GradScaler.scale")

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        """Divide all gradients by scale. Detect inf/nan."""
        raise NotImplementedError("Implement GradScaler.unscale_")

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Call optimizer.step() only if no inf/nan found."""
        raise NotImplementedError("Implement GradScaler.step")

    def update(self) -> None:
        """Update scale factor based on whether inf/nan was found."""
        raise NotImplementedError("Implement GradScaler.update")

    def get_scale(self) -> float:
        """Return current scale factor."""
        raise NotImplementedError("Implement GradScaler.get_scale")


def mixed_precision_train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data: torch.Tensor,
    target: torch.Tensor,
    loss_fn,
    scaler: GradScaler,
) -> float:
    """
    Perform one mixed-precision training step.

    Returns:
        The unscaled loss value as a Python float.
    """
    raise NotImplementedError("Implement mixed_precision_train_step")
