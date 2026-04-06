"""
Solution for Exercise 04: Dual-Clip PPO Loss
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
        eps_clip_c: Dual-clip lower bound (must be > 1.0). Used only when
                    advantage < 0.

    Returns:
        Tuple of:
          - loss: Per-token dual-clip PPO loss, same shape as log_probs.
          - clip_fraction: Fraction of tokens where standard clipping was active.
    """
    assert eps_clip_c > 1.0, f"eps_clip_c must be > 1.0, got {eps_clip_c}"

    # Step 1: Probability ratio
    ratio = np.exp(log_probs - old_log_probs)

    # Step 2: Unclipped surrogate loss
    pg_losses1 = -ratio * advantages

    # Step 3: Standard clipped surrogate loss
    clipped_ratio = np.clip(ratio, 1.0 - eps_clip, 1.0 + eps_clip)
    pg_losses2 = -clipped_ratio * advantages

    # Step 4: Standard PPO (pessimistic -- take the max = worse loss)
    clip_pg_losses1 = np.maximum(pg_losses1, pg_losses2)

    # Step 5: Dual-clip extension
    pg_losses3 = -eps_clip_c * advantages
    # Where advantage < 0, take min(L3, L_clip) to prevent over-penalizing
    clip_pg_losses2 = np.minimum(pg_losses3, clip_pg_losses1)
    loss = np.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    # Step 6: Clip fraction
    clipfrac = float(np.mean(pg_losses2 > pg_losses1))

    return loss, clipfrac
