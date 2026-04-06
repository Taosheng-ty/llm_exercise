"""
Solution for Exercise 02: Gradient Clipping
"""

import torch


def clip_grad_norm(parameters, max_norm: float) -> float:
    """Clip gradient norm across all parameters."""
    parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]

    if len(grads) == 0:
        return 0.0

    total_norm_sq = sum(g.norm(2).item() ** 2 for g in grads)
    total_norm = total_norm_sq ** 0.5

    if total_norm > max_norm:
        clip_coef = max_norm / total_norm
        for g in grads:
            g.mul_(clip_coef)

    return total_norm
