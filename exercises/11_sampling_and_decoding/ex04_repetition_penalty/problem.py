"""Exercise 04: Repetition Penalty (Easy)

Repetition penalty discourages the model from generating tokens that have
already appeared in the context. This is commonly used to reduce repetitive
outputs in open-ended generation.

The penalty is applied asymmetrically:
- For tokens in the context with POSITIVE logits: logit = logit / penalty
  (reduces the logit, making the token less likely)
- For tokens in the context with NEGATIVE logits: logit = logit * penalty
  (makes the logit more negative, also making the token less likely)
- Tokens NOT in the context are unchanged

Implement `apply_repetition_penalty(logits, token_ids, penalty)`:

Args:
    logits: torch.Tensor of shape (batch_size, vocab_size)
    token_ids: list[list[int]] — token IDs in context for each batch element
    penalty: float >= 1.0 — repetition penalty factor (1.0 = no penalty)

Returns:
    torch.Tensor of same shape — logits with penalty applied
"""

import torch


def apply_repetition_penalty(
    logits: torch.Tensor, token_ids: list[list[int]], penalty: float
) -> torch.Tensor:
    # TODO: Implement repetition penalty
    raise NotImplementedError("Implement apply_repetition_penalty")
