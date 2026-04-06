"""Solution for Exercise 05: Beam Search Decoding"""

import torch


def beam_search(
    log_prob_fn,
    beam_width: int,
    max_length: int,
    eos_token_id: int,
    length_penalty: float = 0.0,
) -> list[tuple[list[int], float]]:
    """Beam search decoding.

    Args:
        log_prob_fn: callable(token_ids: Tensor) -> Tensor of log probs (vocab_size,)
        beam_width: number of beams
        max_length: max sequence length including BOS token
        eos_token_id: EOS token ID
        length_penalty: exponent for length normalization

    Returns:
        List of (token_ids, normalized_score) sorted by score descending.
    """
    # Each beam: (token_list, cumulative_log_score)
    active_beams = [([0], 0.0)]  # Start with BOS token (id=0)
    completed_beams = []

    for step in range(max_length - 1):
        if not active_beams:
            break

        all_candidates = []

        for tokens, score in active_beams:
            # Get log probs for next token
            token_tensor = torch.tensor(tokens, dtype=torch.long)
            log_probs = log_prob_fn(token_tensor)  # (vocab_size,)
            vocab_size = log_probs.size(0)

            for token_id in range(vocab_size):
                new_score = score + log_probs[token_id].item()
                new_tokens = tokens + [token_id]
                all_candidates.append((new_tokens, new_score))

        # Sort by cumulative score (descending) and keep top beam_width
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        all_candidates = all_candidates[:beam_width]

        # Separate completed and active beams
        active_beams = []
        for tokens, score in all_candidates:
            if tokens[-1] == eos_token_id:
                completed_beams.append((tokens, score))
            else:
                active_beams.append((tokens, score))

    # If we ran out of steps, add remaining active beams as completed
    for tokens, score in active_beams:
        completed_beams.append((tokens, score))

    # Apply length normalization
    def normalize_score(tokens, score):
        length = len(tokens)
        if length_penalty == 0.0:
            return score
        return score / (length ** length_penalty)

    scored = [(tokens, normalize_score(tokens, score)) for tokens, score in completed_beams]
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:beam_width]
