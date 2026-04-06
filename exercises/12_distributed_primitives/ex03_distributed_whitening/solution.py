"""
Solution for Exercise 03: Distributed Advantage Whitening
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
    """
    # Step 1: Compute local statistics per worker
    local_sums = []
    local_sum_sqs = []
    local_counts = []
    for values, mask in zip(worker_values, worker_masks):
        local_sums.append((values * mask).sum())
        local_sum_sqs.append(((values ** 2) * mask).sum())
        local_counts.append(mask.sum())

    # Step 2: Aggregate across all workers (simulate all-reduce with sum)
    global_sum = sum(local_sums)
    global_sum_sq = sum(local_sum_sqs)
    global_count = sum(local_counts)

    if global_count.item() == 0:
        raise ValueError("Global mask sum is zero across all workers.")

    # Step 3-4: Compute global mean and variance
    global_mean = global_sum / global_count
    global_mean_sq = global_sum_sq / global_count
    global_var = global_mean_sq - global_mean ** 2

    # Step 5: Bessel's correction
    if global_count.item() >= 2:
        bessel_correction = global_count / (global_count - 1)
        global_var = global_var * bessel_correction

    # Step 6-7: Whiten each worker's values
    results = []
    for values in worker_values:
        whitened = (values - global_mean) * torch.rsqrt(global_var + epsilon)
        if not shift_mean:
            whitened = whitened + global_mean
        results.append(whitened)

    return results
