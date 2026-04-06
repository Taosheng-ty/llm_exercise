"""
Solution 04: Entropy Bonus Computation
"""

import torch
import torch.nn.functional as F


def compute_entropy_from_logits(
    logits: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute masked mean entropy from logits.

    Args:
        logits:    (batch, seq_len, vocab_size) model logits.
        loss_mask: (batch, seq_len) binary mask.

    Returns:
        mean_entropy: scalar, average entropy over valid tokens.
    """
    # Numerically stable entropy via log_softmax
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()

    # Per-token entropy: H = -sum(p * log(p), dim=-1)
    per_token_entropy = -(probs * log_probs).sum(dim=-1)  # (batch, seq_len)

    # Masked mean
    num_valid = torch.clamp_min(loss_mask.sum(), 1.0)
    mean_entropy = (per_token_entropy * loss_mask).sum() / num_valid

    return mean_entropy


def compute_loss_with_entropy_bonus(
    policy_loss: torch.Tensor,
    logits: torch.Tensor,
    loss_mask: torch.Tensor,
    entropy_coef: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine policy loss with entropy bonus.

    Args:
        policy_loss: scalar policy gradient loss.
        logits:      (batch, seq_len, vocab_size) model logits.
        loss_mask:   (batch, seq_len) binary mask.
        entropy_coef: weight for entropy bonus.

    Returns:
        total_loss:   scalar = policy_loss - entropy_coef * mean_entropy.
        mean_entropy: scalar, the computed mean entropy (for logging).
    """
    mean_entropy = compute_entropy_from_logits(logits, loss_mask)
    # Subtract entropy (maximizing entropy = minimizing negative entropy)
    total_loss = policy_loss - entropy_coef * mean_entropy
    return total_loss, mean_entropy
