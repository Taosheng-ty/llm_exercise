"""
Solution 06: Reward Normalization with Running Statistics
"""

import torch


class RewardNormalizer:
    """Normalizes rewards using running mean and variance with EMA.

    Args:
        momentum: EMA momentum (higher = more weight on new data).
        eps: small constant for numerical stability.
    """

    def __init__(self, momentum: float = 0.1, eps: float = 1e-8):
        self.running_mean: float = 0.0
        self.running_var: float = 1.0
        self.momentum = momentum
        self.eps = eps
        self.initialized = False

    def update_and_normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        """Update running statistics and return normalized rewards.

        Args:
            rewards: (batch,) or (batch, seq_len) reward tensor.

        Returns:
            normalized: same shape as rewards, normalized using running stats.
        """
        # Compute batch statistics
        batch_mean = rewards.mean().item()
        batch_var = rewards.var().item() if rewards.numel() > 1 else 0.0

        if not self.initialized:
            # First batch: initialize directly
            self.running_mean = batch_mean
            self.running_var = batch_var if batch_var > 0 else 1.0
            self.initialized = True
        else:
            # EMA update
            m = self.momentum
            self.running_mean = (1 - m) * self.running_mean + m * batch_mean
            self.running_var = (1 - m) * self.running_var + m * batch_var

        return self.normalize(rewards)

    def normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalize rewards using current running stats (no update).

        Args:
            rewards: reward tensor of any shape.

        Returns:
            normalized rewards.
        """
        std = (self.running_var + self.eps) ** 0.5
        return (rewards - self.running_mean) / std
