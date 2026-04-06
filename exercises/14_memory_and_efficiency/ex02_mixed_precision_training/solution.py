"""
Solution for Exercise 02: Mixed Precision Training
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
        self._scale = init_scale
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._growth_tracker = 0
        self._found_inf = False

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Multiply loss by current scale factor."""
        return loss * self._scale

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        """Divide all gradients by scale. Detect inf/nan."""
        self._found_inf = False
        inv_scale = 1.0 / self._scale
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.mul_(inv_scale)
                    if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                        self._found_inf = True

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Call optimizer.step() only if no inf/nan found."""
        if not self._found_inf:
            optimizer.step()

    def update(self) -> None:
        """Update scale factor based on whether inf/nan was found."""
        if self._found_inf:
            self._scale *= self._backoff_factor
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self._growth_interval:
                self._scale *= self._growth_factor
                self._growth_tracker = 0

    def get_scale(self) -> float:
        """Return current scale factor."""
        return self._scale


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
    """
    optimizer.zero_grad()

    # Forward pass in fp16
    data_fp16 = data.half()
    with torch.no_grad():
        pass  # no-op, just illustrating context
    # Actually do the forward in fp16
    # We need the model to also operate in fp16 for this to be real mixed precision,
    # but for simplicity we cast input and compute loss
    output = model(data_fp16)
    loss = loss_fn(output.float(), target)

    # Scale loss and backward
    scaled_loss = scaler.scale(loss)
    scaled_loss.backward()

    # Unscale gradients, check for inf/nan
    scaler.unscale_(optimizer)

    # Step (skips if inf/nan)
    scaler.step(optimizer)

    # Update scale factor
    scaler.update()

    return loss.item()
