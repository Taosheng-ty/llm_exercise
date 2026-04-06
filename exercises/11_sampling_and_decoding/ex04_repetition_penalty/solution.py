"""Solution for Exercise 04: Repetition Penalty"""

import torch


def apply_repetition_penalty(
    logits: torch.Tensor, token_ids: list[list[int]], penalty: float
) -> torch.Tensor:
    """Apply repetition penalty to logits for tokens already in context.

    Args:
        logits: (batch_size, vocab_size) raw logits
        token_ids: list of token ID lists, one per batch element
        penalty: penalty factor >= 1.0

    Returns:
        Logits with repetition penalty applied.
    """
    result = logits.clone()

    for i, ids in enumerate(token_ids):
        if not ids:
            continue
        unique_ids = list(set(ids))
        token_indices = torch.tensor(unique_ids, dtype=torch.long, device=logits.device)

        # Gather the logits for the context tokens
        selected_logits = result[i, token_indices]

        # Apply asymmetric penalty
        positive_mask = selected_logits > 0
        selected_logits[positive_mask] = selected_logits[positive_mask] / penalty
        selected_logits[~positive_mask] = selected_logits[~positive_mask] * penalty

        result[i, token_indices] = selected_logits

    return result
