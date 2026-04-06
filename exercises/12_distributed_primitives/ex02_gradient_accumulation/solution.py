"""
Solution for Exercise 02: Gradient Accumulation
"""

import torch
import torch.nn as nn
from typing import Callable


def train_with_gradient_accumulation(
    model: nn.Module,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    data_batches: list[tuple[torch.Tensor, torch.Tensor]],
    accumulation_steps: int,
) -> list[float]:
    """
    Train model using gradient accumulation.
    """
    losses = []
    optimizer.zero_grad()

    for i, (inputs, targets) in enumerate(data_batches):
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()
        losses.append(scaled_loss.item())

        # Step every accumulation_steps or at the very last batch
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(data_batches):
            optimizer.step()
            optimizer.zero_grad()

    return losses
