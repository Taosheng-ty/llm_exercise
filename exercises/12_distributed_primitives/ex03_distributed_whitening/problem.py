"""
Exercise 03: Distributed Advantage Whitening (Medium, PyTorch)

Implement distributed advantage whitening: given advantages and masks spread
across N virtual workers (simulated as lists of tensors), compute global mean
and std from local statistics, then normalize each worker's advantages.

This is a key operation in RLHF/PPO training where advantages must be
normalized using statistics computed across ALL workers, not just local data.

Reference: slime distributed_masked_whiten() in distributed_utils.py.

Implement the following function:

    distributed_masked_whiten(
        worker_values: list[torch.Tensor],
        worker_masks: list[torch.Tensor],
        shift_mean: bool = True,
        epsilon: float = 1e-8,
    ) -> list[torch.Tensor]

Args:
    worker_values: List of N tensors, one per worker. Each has arbitrary shape.
    worker_masks: List of N tensors (same shapes), binary masks (0 or 1).
    shift_mean: If True, output is zero-mean. If False, add global_mean back.
    epsilon: Small constant for numerical stability.

Returns:
    List of N tensors, each whitened using the GLOBAL mean and std computed
    from all workers' masked values.

The algorithm:
    1. Each worker computes local_sum, local_sum_sq, local_count from its
       values and mask.
    2. Aggregate (sum) these across all workers to get global statistics.
    3. Compute global_mean = global_sum / global_count.
    4. Compute global_var = global_sum_sq / global_count - global_mean^2.
    5. Apply Bessel's correction: if count >= 2, var *= count / (count - 1).
    6. Whiten: (values - global_mean) / sqrt(global_var + epsilon).
    7. If not shift_mean, add global_mean back.
"""

import torch


def distributed_masked_whiten(
    worker_values: list[torch.Tensor],
    worker_masks: list[torch.Tensor],
    shift_mean: bool = True,
    epsilon: float = 1e-8,
) -> list[torch.Tensor]:
    """
    Whiten values across virtual workers using global masked statistics.

    TODO: Implement this function.
    """
    raise NotImplementedError("Implement distributed_masked_whiten")
