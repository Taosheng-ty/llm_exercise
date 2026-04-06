"""
Exercise 04: Entropy Bonus Computation
========================================

Implement entropy computation from logits and add it as a bonus to policy loss.

Given:
- logits:    (batch, seq_len, vocab_size) unnormalized log-probabilities
- loss_mask: (batch, seq_len) binary mask for valid tokens

Compute:
1. Convert logits to probabilities: p = softmax(logits, dim=-1)
2. Compute per-token entropy: H = -sum(p * log(p), dim=-1)
   (Use log_softmax for numerical stability)
3. Compute masked mean entropy across valid tokens
4. Combine with policy loss: total_loss = policy_loss - entropy_coef * mean_entropy

The entropy bonus encourages exploration by penalizing overly confident policies.

Reference: slime/utils/ppo_utils.py :: compute_entropy_from_logits()
           slime/backends/megatron_utils/loss.py :: policy_loss_function() (entropy_coef)
"""

import torch


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
    # TODO: Implement entropy computation
    # Hint 1: log_probs = F.log_softmax(logits, dim=-1)
    # Hint 2: probs = log_probs.exp()
    # Hint 3: entropy = -(probs * log_probs).sum(dim=-1)
    # Hint 4: Apply mask and average
    raise NotImplementedError("Implement compute_entropy_from_logits")


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
    # TODO: Combine policy loss with entropy bonus
    raise NotImplementedError("Implement compute_loss_with_entropy_bonus")
