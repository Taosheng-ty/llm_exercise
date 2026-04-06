"""
Exercise 01: Pass@k Metric Estimation

In code generation evaluation (e.g., HumanEval, MBPP), we generate multiple
candidate solutions per problem and want to estimate the probability that at
least one of k randomly chosen samples is correct. This is the pass@k metric.

In RL-based LLM training for code generation, pass@k serves as both an
evaluation metric and a reward signal. It measures whether any of k sampled
solutions passes the test cases, providing the training signal that drives
policy improvement.

Formula:
    pass@k = 1 - C(n-c, k) / C(n, k)

where:
    n = total number of samples generated for the problem
    c = number of correct samples
    C(a, b) = binomial coefficient "a choose b"

A numerically stable way to compute this avoids large factorials:
    pass@k = 1 - prod_{i=n-c+1}^{n} (1 - k/i)
    (when n-c >= k; otherwise pass@k = 1.0)

Reference: slime/utils/metric_utils.py - compute_pass_rate(), _estimate_pass_at_k()
"""

import numpy as np


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Estimate pass@k for a single problem.

    Args:
        n: Total number of samples generated for the problem.
        c: Number of correct samples among the n samples.
        k: Number of samples to select (the k in pass@k).

    Returns:
        The estimated pass@k probability (float between 0.0 and 1.0).

    Edge cases:
        - If c == 0, return 0.0
        - If c == n, return 1.0
        - If k > n, return 1.0 if c > 0, else 0.0

    TODO: Implement this function using the numerically stable formula.
    """
    raise NotImplementedError("Implement estimate_pass_at_k")


def estimate_pass_at_k_batch(
    num_samples: list[int],
    num_correct: list[int],
    k: int,
) -> np.ndarray:
    """
    Estimate pass@k for multiple problems.

    Args:
        num_samples: List of total samples per problem.
        num_correct: List of correct samples per problem.
        k: The k in pass@k.

    Returns:
        numpy array of pass@k estimates, one per problem.

    TODO: Implement by calling estimate_pass_at_k for each problem.
    """
    raise NotImplementedError("Implement estimate_pass_at_k_batch")


def compute_pass_rates(
    flat_rewards: list[float],
    group_size: int,
) -> dict[str, float]:
    """
    Compute pass@1, pass@2, pass@4, ... pass@group_size for a set of problems.

    Given a flat list of binary rewards (1.0 for correct, 0.0 for incorrect),
    where every `group_size` consecutive entries belong to the same problem,
    compute the average pass@k across all problems for each k that is a power of 2
    up to and including group_size.

    Args:
        flat_rewards: Flat list of 0.0/1.0 rewards. Length must be divisible by group_size.
        group_size: Number of samples per problem. Must be >= 2.

    Returns:
        Dictionary mapping "pass@k" -> average pass@k across all problems.
        Returns empty dict if group_size < 2.

    Example:
        flat_rewards = [1, 0, 1, 0, 0, 0, 1, 1]  # 2 problems, 4 samples each
        compute_pass_rates(flat_rewards, group_size=4)
        # Returns {"pass@1": ..., "pass@2": ..., "pass@4": ...}

    TODO: Implement this function.
    """
    raise NotImplementedError("Implement compute_pass_rates")
