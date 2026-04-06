"""Exercise 05: FSM-Based Constrained Decoding (Medium, PyTorch)

When using LLMs for structured output (e.g., choosing from a fixed set of
options, generating valid JSON keys, etc.), constrained decoding ensures the
model can only output tokens that lead to valid completions.

The approach:
1. Build a Finite State Machine (FSM) / trie from the set of valid choices
2. At each decoding step, determine which tokens are valid next tokens
   given the current FSM state
3. Mask out invalid tokens in the logit vector before sampling

FSM structure (trie-based):
    - Each state is a dict mapping token_id -> next_state_id
    - State 0 is the start state
    - A special key "is_terminal" marks states where a valid choice is complete
    - States: dict[int, dict] where each state maps token_ids to next states

Implement:
    - build_fsm_from_choices: build a trie FSM from a list of string choices
    - get_valid_token_mask: get a binary mask of valid next tokens
    - constrained_sample: sample a token from only valid positions
"""

import torch


def build_fsm_from_choices(
    choices: list[str], vocab: dict[str, int]
) -> dict[int, dict]:
    """Build a trie-based FSM from a list of valid string choices.

    Each choice is tokenized character-by-character using the vocab.
    The FSM is a dict of states, where each state maps token_ids to next state IDs.

    Args:
        choices: list of valid output strings (e.g., ["yes", "no", "maybe"])
        vocab: mapping from single characters to token IDs

    Returns:
        FSM as dict[state_id, state_dict] where state_dict maps
        token_id (int) -> next_state_id (int), and may contain
        "is_terminal" -> True for accepting states.

    Example:
        choices=["ab", "ac"], vocab={"a":0, "b":1, "c":2}
        Returns: {
            0: {0: 1},           # start: 'a' -> state 1
            1: {1: 2, 2: 3},     # after 'a': 'b' -> state 2, 'c' -> state 3
            2: {"is_terminal": True},  # "ab" complete
            3: {"is_terminal": True},  # "ac" complete
        }
    """
    # TODO: build the trie
    raise NotImplementedError("Implement build_fsm_from_choices")


def get_valid_token_mask(
    fsm: dict[int, dict], state_id: int, vocab_size: int
) -> torch.Tensor:
    """Get a binary mask of valid next tokens given current FSM state.

    Args:
        fsm: the FSM from build_fsm_from_choices
        state_id: current state in the FSM
        vocab_size: total vocabulary size

    Returns:
        Boolean tensor of shape (vocab_size,).
        True at positions corresponding to valid next tokens, False elsewhere.
        If state_id is terminal or not in FSM, returns all-False mask.
    """
    # TODO: look up valid transitions from fsm[state_id]
    raise NotImplementedError("Implement get_valid_token_mask")


def constrained_sample(
    logits: torch.Tensor, valid_mask: torch.Tensor, temperature: float = 1.0
) -> int:
    """Sample a single token from logits, restricted to valid positions.

    Args:
        logits: 1D tensor of shape (vocab_size,) - raw logits
        valid_mask: boolean tensor of shape (vocab_size,) - True for valid tokens
        temperature: sampling temperature (>0)

    Returns:
        Integer token ID sampled from the valid positions.
        If no valid tokens, return -1.
    """
    # TODO:
    # 1. If no valid tokens, return -1
    # 2. Set invalid positions to -inf
    # 3. Apply temperature scaling
    # 4. Convert to probabilities with softmax
    # 5. Sample from the distribution
    raise NotImplementedError("Implement constrained_sample")
