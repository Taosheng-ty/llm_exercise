"""
Exercise 06: Reward Normalization with Running Statistics
==========================================================

Implement a reward normalizer that maintains running mean and variance
using exponential moving average (EMA) and normalizes incoming rewards.

Given a stream of reward batches, the normalizer should:
1. Maintain running_mean and running_var via EMA
2. Normalize: (rewards - running_mean) / sqrt(running_var + eps)
3. Handle edge cases: first batch, zero variance, empty batches

This is commonly used in RL training to stabilize reward scales across
different tasks or reward functions.

Reference: Common pattern in RL training loops; slime normalizes advantages
           via distributed_masked_whiten() in loss.py
"""

import torch


class RewardNormalizer:
    """Normalizes rewards using running mean and variance with EMA.

    Args:
        momentum: EMA momentum (higher = more weight on new data).
        eps: small constant for numerical stability.
    """

    def __init__(self, momentum: float = 0.1, eps: float = 1e-8):
        # TODO: Initialize running statistics
        # self.running_mean = ...
        # self.running_var = ...
        # self.momentum = momentum
        # self.eps = eps
        # self.initialized = False
        raise NotImplementedError("Implement __init__")

    def update_and_normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        """Update running statistics and return normalized rewards.

        Args:
            rewards: (batch,) or (batch, seq_len) reward tensor.

        Returns:
            normalized: same shape as rewards, normalized using running stats.
        """
        # TODO: Implement
        # 1. Compute batch mean and var
        # 2. If first batch, initialize running stats; else EMA update
        # 3. Normalize using running stats
        raise NotImplementedError("Implement update_and_normalize")

    def normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalize rewards using current running stats (no update).

        Args:
            rewards: reward tensor of any shape.

        Returns:
            normalized rewards.
        """
        # TODO: Normalize without updating stats
        raise NotImplementedError("Implement normalize")
