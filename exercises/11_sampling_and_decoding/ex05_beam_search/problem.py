"""Exercise 05: Beam Search Decoding (Hard)

Beam search maintains multiple candidate sequences (beams) at each decoding
step, expanding each beam with all possible next tokens and keeping only the
top beam_width candidates overall. Beam search finds higher-probability sequences than
greedy decoding by exploring multiple candidates in parallel. It is widely used in LLM
applications like translation and summarization where output quality matters more than
diversity, and is the standard decoding method for non-creative generation tasks.

Implement `beam_search(log_prob_fn, beam_width, max_length, eos_token_id, length_penalty)`:

The function should:
1. Start with a single beam: [BOS] (token 0) with score 0.0
2. At each step, expand each active beam by all vocab tokens
3. Score each expansion: beam_score + log_prob(next_token | beam_tokens)
4. Keep the top beam_width candidates
5. When a beam generates EOS, mark it as complete
6. Continue until all beams are complete or max_length is reached
7. Apply length normalization: final_score = score / (length ** length_penalty)
8. Return the top beam_width sequences sorted by normalized score (best first)

Args:
    log_prob_fn: callable(token_ids: torch.Tensor) -> torch.Tensor
        Given token IDs of shape (seq_len,), returns log probabilities
        of shape (vocab_size,) for the next token.
    beam_width: int — number of beams to maintain
    max_length: int — maximum sequence length (including BOS)
    eos_token_id: int — end of sequence token ID
    length_penalty: float — length normalization exponent (0 = no penalty)

Returns:
    list[tuple[list[int], float]] — list of (token_ids, normalized_score)
        sorted by normalized score descending, length <= beam_width
"""

import torch


def beam_search(
    log_prob_fn,
    beam_width: int,
    max_length: int,
    eos_token_id: int,
    length_penalty: float = 0.0,
) -> list[tuple[list[int], float]]:
    # TODO: Implement beam search decoding
    raise NotImplementedError("Implement beam_search")
