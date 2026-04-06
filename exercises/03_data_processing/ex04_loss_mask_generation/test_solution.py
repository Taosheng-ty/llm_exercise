"""Tests for Exercise 04: Loss Mask Generation"""

import pytest
from .solution import generate_loss_mask, get_response_lengths

# Constants for test token IDs
IM_START = 100
IM_END = 101
NEWLINE = 10
ASSISTANT_IDS = [50]  # "assistant" as a single token
USER_IDS = [20]       # "user" as a single token
SYSTEM_IDS = [30]     # "system" as a single token


def _make_turn(role_ids, content_ids):
    """Helper: build a turn: <|im_start|> role \\n content <|im_end|> \\n"""
    return [IM_START] + role_ids + [NEWLINE] + content_ids + [IM_END, NEWLINE]


class TestGenerateLossMask:
    def test_single_user_turn_all_zeros(self):
        tokens = _make_turn(USER_IDS, [1, 2, 3])
        mask = generate_loss_mask(tokens, IM_START, IM_END, ASSISTANT_IDS, NEWLINE)
        assert mask == [0] * len(tokens)

    def test_single_assistant_turn(self):
        tokens = _make_turn(ASSISTANT_IDS, [4, 5, 6])
        mask = generate_loss_mask(tokens, IM_START, IM_END, ASSISTANT_IDS, NEWLINE)
        # Header: im_start(0), assistant(0), newline(0) -> all 0
        # Content: 4(1), 5(1), 6(1)
        # Footer: im_end(0), newline(0)
        expected = [0, 0, 0, 1, 1, 1, 0, 0]
        assert mask == expected

    def test_multi_turn_user_then_assistant(self):
        user_turn = _make_turn(USER_IDS, [1, 2, 3])
        asst_turn = _make_turn(ASSISTANT_IDS, [4, 5])
        tokens = user_turn + asst_turn
        mask = generate_loss_mask(tokens, IM_START, IM_END, ASSISTANT_IDS, NEWLINE)
        # User turn: all zeros (7 tokens)
        # Assistant turn: 3 header zeros, 2 content ones, 2 footer zeros
        expected = [0] * len(user_turn) + [0, 0, 0, 1, 1, 0, 0]
        assert mask == expected

    def test_multi_turn_multiple_assistant_turns(self):
        """Two user-assistant pairs; both assistant turns should be masked in."""
        turn1_user = _make_turn(USER_IDS, [1])
        turn1_asst = _make_turn(ASSISTANT_IDS, [2, 3])
        turn2_user = _make_turn(USER_IDS, [4])
        turn2_asst = _make_turn(ASSISTANT_IDS, [5, 6, 7])
        tokens = turn1_user + turn1_asst + turn2_user + turn2_asst
        mask = generate_loss_mask(tokens, IM_START, IM_END, ASSISTANT_IDS, NEWLINE)

        # Count masked-in tokens: should be 2 + 3 = 5
        assert sum(mask) == 5

    def test_system_user_assistant(self):
        sys_turn = _make_turn(SYSTEM_IDS, [11, 12])
        user_turn = _make_turn(USER_IDS, [1, 2])
        asst_turn = _make_turn(ASSISTANT_IDS, [3, 4, 5])
        tokens = sys_turn + user_turn + asst_turn
        mask = generate_loss_mask(tokens, IM_START, IM_END, ASSISTANT_IDS, NEWLINE)
        # Only assistant content tokens should be 1
        assert sum(mask) == 3
        # System and user turns should be all zeros
        sys_user_len = len(sys_turn) + len(user_turn)
        assert mask[:sys_user_len] == [0] * sys_user_len

    def test_empty_assistant_content(self):
        tokens = _make_turn(ASSISTANT_IDS, [])
        mask = generate_loss_mask(tokens, IM_START, IM_END, ASSISTANT_IDS, NEWLINE)
        assert mask == [0] * len(tokens)

    def test_multi_token_assistant_role(self):
        """Test when 'assistant' is tokenized as multiple tokens."""
        multi_asst_ids = [50, 51]  # e.g., "assis" + "tant"
        tokens = [IM_START] + multi_asst_ids + [NEWLINE, 7, 8, IM_END, NEWLINE]
        mask = generate_loss_mask(tokens, IM_START, IM_END, multi_asst_ids, NEWLINE)
        # header: im_start, 50, 51, newline -> 4 zeros
        # content: 7, 8 -> 2 ones
        # footer: im_end, newline -> 2 zeros
        expected = [0, 0, 0, 0, 1, 1, 0, 0]
        assert mask == expected


class TestGetResponseLengths:
    def test_basic(self):
        masks = [[0, 0, 1, 1, 0], [0, 0, 0]]
        assert get_response_lengths(masks) == [3, 0]

    def test_all_ones(self):
        masks = [[1, 1, 1]]
        assert get_response_lengths(masks) == [3]

    def test_no_ones(self):
        masks = [[0, 0, 0, 0]]
        assert get_response_lengths(masks) == [0]

    def test_multiple_samples(self):
        masks = [
            [0, 0, 0, 1, 1, 1, 0, 0],  # first 1 at index 3 -> length = 8 - 3 = 5
            [0, 1, 0, 0],               # first 1 at index 1 -> length = 4 - 1 = 3
            [0, 0],                      # no 1s -> 0
        ]
        assert get_response_lengths(masks) == [5, 3, 0]

    def test_empty_mask(self):
        masks = [[]]
        assert get_response_lengths(masks) == [0]
