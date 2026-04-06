"""Solution for Exercise 05: FSM-Based Constrained Decoding"""

import torch


def build_fsm_from_choices(
    choices: list[str], vocab: dict[str, int]
) -> dict[int, dict]:
    """Build a trie-based FSM from a list of valid string choices."""
    fsm = {0: {}}
    next_state_id = 1

    for choice in choices:
        current_state = 0
        for char in choice:
            token_id = vocab[char]
            if token_id not in fsm[current_state]:
                # Create a new state
                fsm[current_state][token_id] = next_state_id
                fsm[next_state_id] = {}
                next_state_id += 1
            current_state = fsm[current_state][token_id]
        # Mark end of valid choice
        fsm[current_state]["is_terminal"] = True

    return fsm


def get_valid_token_mask(
    fsm: dict[int, dict], state_id: int, vocab_size: int
) -> torch.Tensor:
    """Get a binary mask of valid next tokens given current FSM state."""
    mask = torch.zeros(vocab_size, dtype=torch.bool)

    if state_id not in fsm:
        return mask

    state = fsm[state_id]
    # If this is a terminal state, no more tokens are valid
    if state.get("is_terminal", False) and len([k for k in state if k != "is_terminal"]) == 0:
        return mask

    for key in state:
        if key == "is_terminal":
            continue
        if isinstance(key, int) and 0 <= key < vocab_size:
            mask[key] = True

    return mask


def constrained_sample(
    logits: torch.Tensor, valid_mask: torch.Tensor, temperature: float = 1.0
) -> int:
    """Sample a single token from logits, restricted to valid positions."""
    if not valid_mask.any():
        return -1

    # Mask invalid tokens to -inf
    masked_logits = logits.clone()
    masked_logits[~valid_mask] = float("-inf")

    # Apply temperature
    scaled_logits = masked_logits / temperature

    # Sample
    probs = torch.softmax(scaled_logits, dim=-1)
    token_id = torch.multinomial(probs, num_samples=1).item()
    return token_id
