"""
Exercise 02: Gradient Accumulation (Medium, PyTorch)

Implement gradient accumulation over micro-batches. In distributed training,
when the effective batch size is too large for GPU memory, we split it into
micro-batches and accumulate gradients before calling optimizer.step().

LLM training requires large effective batch sizes for stable convergence, but
single-GPU memory can't hold large batches of long sequences. Gradient
accumulation simulates larger batches by accumulating gradients over multiple
micro-batches before updating, enabling effective batch sizes of thousands of
sequences on limited hardware.

Implement the following function:

    train_with_gradient_accumulation(
        model: nn.Module,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        data_batches: list[tuple[torch.Tensor, torch.Tensor]],
        accumulation_steps: int,
    ) -> list[float]

Args:
    model: A PyTorch model (e.g., a simple linear layer).
    loss_fn: A loss function that takes (predictions, targets) and returns a scalar.
    optimizer: An optimizer (e.g., SGD or Adam).
    data_batches: A list of (input, target) micro-batch tuples.
    accumulation_steps: Number of micro-batches to accumulate before stepping.

Returns:
    A list of loss values (one per micro-batch, scaled by 1/accumulation_steps).

Key requirements:
    1. Scale each micro-batch loss by 1/accumulation_steps before .backward().
       Note: when using mean-reduction losses like nn.MSELoss() (default reduction='mean'),
       each micro-batch loss is already averaged over the micro-batch samples. Dividing
       by accumulation_steps makes the accumulated gradient match a single large-batch
       gradient (mean-of-means).
    2. Call optimizer.step() and optimizer.zero_grad() every accumulation_steps batches.
    3. If the total number of batches is not divisible by accumulation_steps,
       perform a final step with the remaining accumulated gradients.
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

    TODO: Implement this function.
    """
    raise NotImplementedError("Implement train_with_gradient_accumulation")
