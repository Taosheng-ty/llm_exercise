"""
Exercise 02: Gradient Clipping

Difficulty: Easy
Framework: PyTorch

Background:
    Gradient clipping prevents exploding gradients during training by scaling
    down all gradients when the global gradient norm exceeds a threshold.
    This is essential for stable LLM training — large language models are prone
    to occasional gradient spikes (especially during RL fine-tuning) that can
    cause loss divergence and corrupt the model. Clipping by global norm is the
    standard approach used in virtually all LLM training frameworks.

    The algorithm:
    1. Compute global gradient norm: sqrt(sum of squared norms of all param grads)
    2. If the norm exceeds max_norm, scale all gradients by (max_norm / norm)
    3. Return the original (unclipped) norm for logging

    This is equivalent to torch.nn.utils.clip_grad_norm_ but you should
    implement it from scratch.

Args:
    parameters: iterable of torch.nn.Parameter (model.parameters())
    max_norm: float - maximum allowed gradient norm

Returns:
    total_norm: float - the original gradient norm before clipping
"""

import torch


def clip_grad_norm(parameters, max_norm: float) -> float:
    """Clip gradient norm across all parameters.

    Args:
        parameters: iterable of Parameters with .grad attributes
        max_norm: maximum allowed gradient norm

    Returns:
        The total gradient norm before clipping (as a Python float).
    """
    # TODO: Implement gradient norm clipping
    # Hint 1: Collect all parameter gradients (skip params where grad is None)
    # Hint 2: Compute per-param norm: param.grad.norm(2)
    # Hint 3: Global norm = sqrt(sum of squared per-param norms)
    # Hint 4: If global norm > max_norm, scale each grad by (max_norm / global_norm)
    raise NotImplementedError("Implement clip_grad_norm")
