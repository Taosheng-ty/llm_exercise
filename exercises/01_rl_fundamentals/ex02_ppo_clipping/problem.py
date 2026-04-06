"""
Exercise 02: PPO Clipped Surrogate Objective

Difficulty: Medium

Background:
    PPO prevents destructively large policy updates by clipping the probability
    ratio between new and old policies. The clipped surrogate objective is:

        ratio = exp(log_probs_new - log_probs_old)
        unclipped = ratio * advantages
        clipped   = clip(ratio, 1 - eps_clip, 1 + eps_clip) * advantages
        loss      = -min(unclipped, clipped)    (per-token, we negate for gradient ascent)

    The clip prevents the ratio from moving too far from 1.0, stabilizing training.

    Reference: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
    Also see: slime/utils/ppo_utils.py compute_policy_loss()

Args:
    log_probs_new: numpy array of shape (N,) - log probs under current policy
    log_probs_old: numpy array of shape (N,) - log probs under old policy
    advantages:    numpy array of shape (N,) - advantage estimates
    eps_clip:      float - clipping epsilon (typically 0.2)

Returns:
    policy_loss: numpy array of shape (N,) - per-token clipped policy loss
    clip_fraction: float - fraction of samples where the ratio falls outside [1-eps, 1+eps]
"""

import numpy as np


def compute_policy_loss(
    log_probs_new: np.ndarray,
    log_probs_old: np.ndarray,
    advantages: np.ndarray,
    eps_clip: float,
) -> tuple[np.ndarray, float]:
    """Compute PPO clipped surrogate policy loss.

    See module docstring for full details.
    """
    # TODO: Implement PPO clipped surrogate objective
    # Hint 1: ratio = exp(log_probs_new - log_probs_old)
    # Hint 2: Compute both unclipped and clipped objectives
    # Hint 3: Take the element-wise maximum (pessimistic bound) then negate
    # Hint 4: clip_fraction = fraction of samples where clipped != unclipped
    raise NotImplementedError("Implement compute_policy_loss")
