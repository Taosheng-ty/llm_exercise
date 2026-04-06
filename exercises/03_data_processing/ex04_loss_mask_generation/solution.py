"""
Exercise 04: Loss Mask Generation for SFT Training - Solution
"""


def generate_loss_mask(
    token_ids: list[int],
    im_start_id: int,
    im_end_id: int,
    assistant_token_ids: list[int],
    newline_id: int,
) -> list[int]:
    """Generate a loss mask for SFT training on multi-turn conversations."""
    n = len(token_ids)
    mask = [0] * n

    # The header for an assistant turn is: im_start + assistant_token_ids + newline
    assistant_header = [im_start_id] + list(assistant_token_ids) + [newline_id]
    header_len = len(assistant_header)

    i = 0
    while i < n:
        if token_ids[i] == im_start_id:
            # Check if this is an assistant turn
            if i + header_len <= n and token_ids[i : i + header_len] == assistant_header:
                # This is an assistant turn
                # Skip the header (mask = 0)
                content_start = i + header_len
                # Find the matching im_end
                j = content_start
                while j < n and token_ids[j] != im_end_id:
                    j += 1
                # Set mask=1 for content tokens
                for k in range(content_start, j):
                    mask[k] = 1
                # im_end and following newline stay 0
                # Advance past im_end and optional newline
                if j < n:
                    j += 1  # skip im_end
                if j < n and token_ids[j] == newline_id:
                    j += 1  # skip newline after im_end
                i = j
            else:
                # Non-assistant turn: skip to im_end
                j = i + 1
                while j < n and token_ids[j] != im_end_id:
                    j += 1
                if j < n:
                    j += 1  # skip im_end
                if j < n and token_ids[j] == newline_id:
                    j += 1  # skip newline after im_end
                i = j
        else:
            i += 1

    return mask


def get_response_lengths(loss_masks: list[list[int]]) -> list[int]:
    """Compute the length of the response portion for each sample."""
    result = []
    for mask in loss_masks:
        if 1 in mask:
            first_one = mask.index(1)
            result.append(len(mask) - first_one)
        else:
            result.append(0)
    return result
