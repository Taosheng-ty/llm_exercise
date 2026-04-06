"""
Solution for Exercise 03: KL Divergence Approximation Methods
"""

import numpy as np


def compute_approx_kl(
    log_probs_new: np.ndarray,
    log_probs_old: np.ndarray,
    kl_type: str,
) -> np.ndarray:
    """Compute approximate KL divergence between two distributions."""
    log_ratio = log_probs_new - log_probs_old

    if kl_type == "k1":
        kl = log_ratio
    elif kl_type == "k2":
        kl = log_ratio ** 2 / 2.0
    elif kl_type == "k3":
        # Non-negative KL approximation from Schulman's blog
        neg_log_ratio = -log_ratio
        kl = np.exp(neg_log_ratio) - 1 - neg_log_ratio
    else:
        raise ValueError(f"Unknown kl_type: {kl_type}")

    return kl
