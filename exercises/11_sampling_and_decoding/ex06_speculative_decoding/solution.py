"""Solution for Exercise 06: Speculative Decoding

Inspired by slime's SpecInfo tracking (spec_accept_token_num, spec_draft_token_num,
spec_verify_ct) in slime/utils/types.py.
"""

import torch


def speculative_decode(
    prefix: list[int],
    draft_model,
    target_model,
    k: int,
    random_seed: int = 42,
) -> tuple[list[int], dict]:
    """Speculative decoding: draft k tokens, verify with target model.

    Args:
        prefix: current token sequence
        draft_model: callable(tokens) -> logits (vocab_size,) for next token
        target_model: callable(tokens) -> logits (seq_len, vocab_size)
        k: number of draft tokens
        random_seed: for reproducibility

    Returns:
        (new_tokens, stats_dict)
    """
    torch.manual_seed(random_seed)

    # Step 1: Draft model generates k tokens autoregressively
    draft_tokens = []
    draft_probs_list = []
    current = list(prefix)

    for _ in range(k):
        draft_logits = draft_model(current)  # (vocab_size,)
        draft_probs = torch.softmax(draft_logits, dim=-1)
        # Sample from draft distribution
        token = torch.multinomial(draft_probs, num_samples=1).item()
        draft_tokens.append(token)
        draft_probs_list.append(draft_probs)
        current.append(token)

    # Step 2: Target model scores all positions in one pass
    # current = prefix + draft_tokens
    target_logits = target_model(current)  # (len(current), vocab_size)
    # We need target probs at positions len(prefix)-1 through len(prefix)+k-1
    # Position len(prefix)-1 predicts token at len(prefix), etc.

    # Step 3: Accept/reject each drafted token
    new_tokens = []
    accepted = 0

    for i in range(k):
        target_pos = len(prefix) - 1 + i  # position in target logits
        target_probs = torch.softmax(target_logits[target_pos], dim=-1)
        draft_probs = draft_probs_list[i]
        drafted_token = draft_tokens[i]

        # Acceptance probability
        p_target = target_probs[drafted_token].item()
        p_draft = draft_probs[drafted_token].item()

        if p_draft == 0:
            accept_prob = 1.0 if p_target > 0 else 0.0
        else:
            accept_prob = min(1.0, p_target / p_draft)

        r = torch.rand(1).item()

        if r < accept_prob:
            # Accept
            new_tokens.append(drafted_token)
            accepted += 1
        else:
            # Reject: resample from adjusted distribution max(0, p_target - p_draft)
            adjusted = torch.clamp(target_probs - draft_probs, min=0.0)
            adj_sum = adjusted.sum()
            if adj_sum > 0:
                adjusted = adjusted / adj_sum
            else:
                # Fallback to target distribution
                adjusted = target_probs
            resampled = torch.multinomial(adjusted, num_samples=1).item()
            new_tokens.append(resampled)
            break

    # Step 4: If all k tokens accepted, sample bonus token from target
    if accepted == k:
        bonus_pos = len(prefix) - 1 + k
        if bonus_pos < target_logits.size(0):
            bonus_probs = torch.softmax(target_logits[bonus_pos], dim=-1)
            bonus_token = torch.multinomial(bonus_probs, num_samples=1).item()
            new_tokens.append(bonus_token)

    stats = {
        "draft_tokens": k,
        "accepted_tokens": accepted,
    }

    return new_tokens, stats
