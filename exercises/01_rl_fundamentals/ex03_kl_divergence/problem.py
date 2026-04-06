"""
Exercise 03: KL Divergence Approximation Methods

Difficulty: Easy

Background:
    In RLHF/PPO training, we monitor how far the current policy has drifted
    from the reference policy using KL divergence approximations. Since computing
    exact KL requires full distributions, we use approximations based on
    log probability ratios.

    Given log_ratio = log(pi_new / pi_old) = log_probs_new - log_probs_old:

    k1 (simple):   KL ~ log_ratio
        - Unbiased but can be negative for individual samples
    k2 (Schulman): KL ~ 0.5 * log_ratio^2
        - Always non-negative, but biased
    k3 (abs/low-var): KL ~ exp(-log_ratio) - 1 + log_ratio
        - Always non-negative, unbiased, low variance
        - Note: uses NEGATIVE log_ratio (i.e., log(pi_old/pi_new))

    Reference: http://joschu.net/blog/kl-approx.html
    Also see: slime/utils/ppo_utils.py compute_approx_kl()

Args:
    log_probs_new: numpy array - log probs under new policy
    log_probs_old: numpy array - log probs under old policy
    kl_type:       str - one of "k1", "k2", "k3"

Returns:
    kl: numpy array of same shape - approximate KL divergence values
"""

import numpy as np


def compute_approx_kl(
    log_probs_new: np.ndarray,
    log_probs_old: np.ndarray,
    kl_type: str,
) -> np.ndarray:
    """Compute approximate KL divergence between two distributions.

    See module docstring for full details.
    """
    # TODO: Implement k1, k2, and k3 KL approximations
    # Hint 1: Start by computing log_ratio = log_probs_new - log_probs_old
    # Hint 2: For k3, negate the log_ratio first, then compute exp(neg_lr) - 1 - neg_lr
    # Hint 3: Raise ValueError for unknown kl_type
    raise NotImplementedError("Implement compute_approx_kl")
