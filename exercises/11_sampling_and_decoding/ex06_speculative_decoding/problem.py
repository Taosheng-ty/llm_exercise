"""Exercise 06: Speculative Decoding (Hard)

Speculative decoding accelerates LLM inference by using a small "draft" model
to propose multiple tokens, then verifying them in parallel with the larger
"target" model. This exploits the fact that verification (one forward pass for
k tokens) is cheaper than k sequential forward passes of the target model.

Inspired by slime's SpecInfo tracking in types.py, which records
spec_accept_token_num, spec_draft_token_num, and spec_verify_ct.

Algorithm (simplified):
1. Draft model generates k tokens autoregressively
2. Target model scores all k+1 positions in one forward pass
3. For each drafted token i (left to right):
   - Compute acceptance probability: min(1, p_target(x_i) / p_draft(x_i))
   - Draw uniform random number r ~ U(0,1)
   - If r < acceptance_prob: ACCEPT the token, continue to next
   - If r >= acceptance_prob: REJECT. Resample from adjusted distribution:
     norm(max(0, p_target(x) - p_draft(x))) and stop accepting further tokens.
4. If all k tokens accepted, sample one bonus token from target model at position k+1.

Implement `speculative_decode(prefix, draft_model, target_model, k, random_seed)`:

Args:
    prefix: list[int] — current token sequence
    draft_model: callable(tokens: list[int]) -> torch.Tensor
        Returns logits of shape (vocab_size,) for next token prediction
    target_model: callable(tokens: list[int]) -> torch.Tensor
        Returns logits of shape (seq_len, vocab_size) for ALL positions
        (i.e., given n tokens, returns n rows of logits)
    k: int — number of draft tokens to propose
    random_seed: int — seed for reproducibility

Returns:
    tuple of:
        - new_tokens: list[int] — accepted (and possibly resampled) tokens
        - stats: dict with keys:
            - "draft_tokens": int — number of tokens drafted (k)
            - "accepted_tokens": int — number of tokens accepted
"""

import torch


def speculative_decode(
    prefix: list[int],
    draft_model,
    target_model,
    k: int,
    random_seed: int = 42,
) -> tuple[list[int], dict]:
    # TODO: Implement speculative decoding
    raise NotImplementedError("Implement speculative_decode")
