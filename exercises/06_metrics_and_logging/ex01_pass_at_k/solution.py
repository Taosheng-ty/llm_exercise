"""
Exercise 01: Pass@k Metric Estimation - Solution

Reference: slime/utils/metric_utils.py - compute_pass_rate(), _estimate_pass_at_k()
"""

import math

import numpy as np


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Estimate pass@k for a single problem using the numerically stable formula.

    pass@k = 1 - prod_{i=n-c+1}^{n} (1 - k/i)
    """
    # Edge cases
    if c == 0:
        return 0.0
    if c >= n:
        return 1.0
    if k > n:
        return 1.0 if c > 0 else 0.0

    # If there aren't enough incorrect samples to fill all k slots,
    # then at least one correct sample must be chosen.
    if n - c < k:
        return 1.0

    # Numerically stable computation: 1 - C(n-c, k) / C(n, k)
    # = 1 - prod_{i=n-c+1}^{n} (1 - k/i)
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def estimate_pass_at_k_batch(
    num_samples: list[int],
    num_correct: list[int],
    k: int,
) -> np.ndarray:
    """
    Estimate pass@k for multiple problems.
    """
    return np.array([
        estimate_pass_at_k(int(n), int(c), k)
        for n, c in zip(num_samples, num_correct)
    ])


def compute_pass_rates(
    flat_rewards: list[float],
    group_size: int,
) -> dict[str, float]:
    """
    Compute pass@1, pass@2, pass@4, ... pass@group_size for a set of problems.
    """
    if group_size < 2:
        return {}

    num_groups = len(flat_rewards) // group_size
    assert len(flat_rewards) == num_groups * group_size

    rewards_of_group = np.array(flat_rewards).reshape(num_groups, group_size)

    # Powers of 2 up to and including group_size: 1, 2, 4, ..., group_size
    pass_rate_ks = [2**i for i in range(int(math.log2(group_size)) + 1)]

    log_dict = {}
    for k in pass_rate_ks:
        num_correct = np.sum(rewards_of_group == 1, axis=1)
        num_samples = np.full(num_groups, group_size)

        pass_k_estimates = estimate_pass_at_k_batch(
            num_samples.tolist(), num_correct.tolist(), k
        )
        log_dict[f"pass@{k}"] = float(np.mean(pass_k_estimates))

    return log_dict
