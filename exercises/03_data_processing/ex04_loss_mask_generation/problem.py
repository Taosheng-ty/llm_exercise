"""
Exercise 04: Loss Mask Generation for SFT Training (Medium)

In Supervised Fine-Tuning (SFT), we train the model to produce assistant responses
but NOT to memorize the user prompts or system instructions. The loss mask controls
which tokens contribute to the training loss:
  - mask = 1: compute loss on this token (assistant response tokens)
  - mask = 0: ignore this token (system/user prompt tokens)

The input is a tokenized conversation with special delimiter tokens that mark where
each role's content begins and ends:

  <|im_start|>system\n...content...<|im_end|>\n
  <|im_start|>user\n...content...<|im_end|>\n
  <|im_start|>assistant\n...content...<|im_end|>\n

Given a flat list of token IDs and the special token IDs, generate the loss mask.

Reference: slime/utils/mask_utils.py MultiTurnLossMaskGenerator
"""


def generate_loss_mask(
    token_ids: list[int],
    im_start_id: int,
    im_end_id: int,
    assistant_token_ids: list[int],
    newline_id: int,
) -> list[int]:
    """Generate a loss mask for SFT training on multi-turn conversations.

    The mask should be 1 for tokens that are part of assistant responses and 0
    for everything else (system messages, user messages, special tokens, delimiters).

    Specifically, for each assistant turn:
    - The <|im_start|> token, "assistant" role tokens, and the newline after the
      role are masked OUT (0). These are the "header" tokens.
    - The actual content tokens of the assistant response are masked IN (1).
    - The <|im_end|> token and following newline are masked OUT (0).

    For all non-assistant turns, everything is masked OUT (0).

    Algorithm:
    1. Scan through token_ids to find each occurrence of im_start_id.
    2. After im_start_id, check if the next tokens match assistant_token_ids
       followed by newline_id. If so, this is an assistant turn.
    3. For assistant turns, set mask=1 for all content tokens between
       the header (im_start + role + newline) and the im_end token.
    4. For all other tokens, mask=0.

    Args:
        token_ids: The full tokenized conversation as a flat list of ints.
        im_start_id: Token ID for <|im_start|>.
        im_end_id: Token ID for <|im_end|>.
        assistant_token_ids: Token IDs for the word "assistant" (could be
            multiple tokens depending on tokenizer, e.g., [78191] or [519, 11144]).
        newline_id: Token ID for "\\n".

    Returns:
        A list of ints (0 or 1) of the same length as token_ids.

    Example:
        >>> # Simplified: im_start=100, im_end=101, assistant=[50], newline=10
        >>> tokens = [100, 20, 10, 1, 2, 3, 101, 10,   # user turn
        ...           100, 50, 10, 4, 5, 6, 101, 10]    # assistant turn
        >>> mask = generate_loss_mask(tokens, 100, 101, [50], 10)
        >>> # mask =  [0,  0,  0, 0, 0, 0,   0,  0,     # user: all 0
        >>> #          0,  0,  0, 1, 1, 1,   0,  0]      # assistant: content=1
    """
    # TODO: Implement this function
    raise NotImplementedError


def get_response_lengths(loss_masks: list[list[int]]) -> list[int]:
    """Compute the length of the response portion for each sample.

    The response length is defined as the number of tokens from the FIRST
    occurrence of mask=1 to the end of the sequence.

    This is useful for filtering out samples where the response was truncated
    or too short.

    Args:
        loss_masks: List of loss mask lists, one per sample.

    Returns:
        List of ints: for each mask, the number of tokens from the first 1
        to the end. Returns 0 if there are no 1s in the mask.

    Example:
        >>> get_response_lengths([[0, 0, 1, 1, 0], [0, 0, 0]])
        [3, 0]
    """
    # TODO: Implement this function
    raise NotImplementedError
