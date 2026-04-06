"""
Exercise 04: Dual-Clip PPO Loss

Standard PPO clips the probability ratio to [1-eps, 1+eps] to prevent too-large
policy updates. Dual-Clip PPO adds an additional lower bound c (c > 1) that
prevents the policy from moving too far when the advantage is negative.

The algorithm:
  1. Compute ratio = exp(log_prob - old_log_prob).
  2. Unclipped loss:   L1 = -ratio * advantage
  3. Standard clipped: L2 = -clip(ratio, 1-eps, 1+eps) * advantage
  4. Standard PPO:     L_clip = max(L1, L2)  [pessimistic bound]
  5. Dual-clip extra:  L3 = -c * advantage
  6. When advantage < 0:
       L_dual = min(L3, L_clip)  [prevents over-penalizing bad actions]
     When advantage >= 0:
       L_dual = L_clip
  7. Return per-token L_dual and the clip fraction.

Reference: slime's compute_policy_loss() with eps_clip_c parameter.

Difficulty: Hard
"""
import numpy as np


def dual_clip_ppo_loss(
    log_probs: np.ndarray,
    old_log_probs: np.ndarray,
    advantages: np.ndarray,
    eps_clip: float = 0.2,
    eps_clip_c: float = 3.0,
) -> tuple[np.ndarray, float]:
    """Compute Dual-Clip PPO policy loss.

    Args:
        log_probs: Current policy log probs, shape (batch,) or (num_tokens,).
        old_log_probs: Old policy log probs, same shape as log_probs.
        advantages: Per-token advantage estimates, same shape as log_probs.
        eps_clip: Standard PPO clip range. Ratio clipped to [1-eps, 1+eps].
        eps_clip_c: Dual-clip lower bound (must be > 1.0; assert this). Used
                    only when advantage < 0.

    Returns:
        Tuple of:
          - loss: Per-token dual-clip PPO loss, same shape as log_probs.
          - clip_fraction: Fraction of tokens where standard clipping was active.
    """
    # TODO: Implement dual-clip PPO loss
    #   1. Compute the probability ratio from log probs.
    #   2. Compute unclipped surrogate loss L1.
    #   3. Compute standard clipped surrogate loss L2.
    #   4. Standard PPO: L_clip = max(L1, L2) -- pessimistic.
    #   5. Dual-clip: L3 = -c * advantage. Where advantage < 0, take
    #      min(L3, L_clip); otherwise keep L_clip.
    #   6. Compute clip fraction = fraction of tokens where L2 > L1.
    raise NotImplementedError
