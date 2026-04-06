"""Solution for Exercise 03: Top-P (Nucleus) Sampling"""

import torch


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Filter logits using nucleus (top-p) sampling.

    Keeps the smallest set of tokens whose cumulative probability >= p.
    Always keeps at least the top-1 token.

    Args:
        logits: (batch_size, vocab_size) raw logits
        p: cumulative probability threshold in (0, 1]

    Returns:
        Filtered logits with low-probability tokens set to -inf.
    """
    # Sort logits descending
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)

    # Compute cumulative probabilities from sorted logits
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Create mask: remove tokens with cumulative probability above the threshold
    # We shift cumulative_probs right by 1 so the token that crosses the threshold is kept
    sorted_mask = cumulative_probs - sorted_probs > p

    # Always keep at least the first token (top-1)
    sorted_mask[:, 0] = False

    # Set masked positions to -inf in sorted space
    sorted_logits[sorted_mask] = float("-inf")

    # Scatter back to original positions
    result = torch.zeros_like(logits)
    result.scatter_(-1, sorted_indices, sorted_logits)
    return result
