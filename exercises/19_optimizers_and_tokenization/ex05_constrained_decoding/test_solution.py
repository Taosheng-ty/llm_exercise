"""Tests for Exercise 05: FSM-Based Constrained Decoding"""

import importlib.util
import os

import pytest
import torch

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
build_fsm_from_choices = _mod.build_fsm_from_choices
get_valid_token_mask = _mod.get_valid_token_mask
constrained_sample = _mod.constrained_sample


@pytest.fixture
def simple_vocab():
    return {chr(c): i for i, c in enumerate(range(ord("a"), ord("z") + 1))}


class TestBuildFsm:
    def test_single_choice(self, simple_vocab):
        """FSM for a single choice should be a linear chain."""
        fsm = build_fsm_from_choices(["ab"], simple_vocab)
        # State 0 -> 'a' -> State 1 -> 'b' -> State 2 (terminal)
        assert simple_vocab["a"] in fsm[0]
        state_after_a = fsm[0][simple_vocab["a"]]
        assert simple_vocab["b"] in fsm[state_after_a]
        state_after_ab = fsm[state_after_a][simple_vocab["b"]]
        assert fsm[state_after_ab].get("is_terminal") is True

    def test_shared_prefix(self, simple_vocab):
        """Choices with shared prefix should share FSM states."""
        fsm = build_fsm_from_choices(["ab", "ac"], simple_vocab)
        # Both start with 'a', so state 0 should have only one transition
        a_id = simple_vocab["a"]
        assert a_id in fsm[0]
        state_after_a = fsm[0][a_id]
        # After 'a', both 'b' and 'c' should be valid
        assert simple_vocab["b"] in fsm[state_after_a]
        assert simple_vocab["c"] in fsm[state_after_a]

    def test_no_shared_prefix(self, simple_vocab):
        """Choices with no shared prefix should have separate branches."""
        fsm = build_fsm_from_choices(["ab", "cd"], simple_vocab)
        assert simple_vocab["a"] in fsm[0]
        assert simple_vocab["c"] in fsm[0]

    def test_start_state_exists(self, simple_vocab):
        """State 0 should always exist."""
        fsm = build_fsm_from_choices(["x"], simple_vocab)
        assert 0 in fsm

    def test_terminal_states_marked(self, simple_vocab):
        """Each choice endpoint should be marked as terminal."""
        fsm = build_fsm_from_choices(["a", "b"], simple_vocab)
        # 'a' leads to a terminal state
        state_a = fsm[0][simple_vocab["a"]]
        assert fsm[state_a].get("is_terminal") is True
        state_b = fsm[0][simple_vocab["b"]]
        assert fsm[state_b].get("is_terminal") is True


class TestGetValidTokenMask:
    def test_start_state_valid_tokens(self, simple_vocab):
        """At start state, only first characters of choices should be valid."""
        fsm = build_fsm_from_choices(["yes", "no"], simple_vocab)
        mask = get_valid_token_mask(fsm, state_id=0, vocab_size=26)
        assert mask[simple_vocab["y"]].item() is True
        assert mask[simple_vocab["n"]].item() is True
        assert mask[simple_vocab["a"]].item() is False

    def test_terminal_state_no_valid(self, simple_vocab):
        """Terminal states with no outgoing transitions should have all-False mask."""
        fsm = build_fsm_from_choices(["a"], simple_vocab)
        terminal = fsm[0][simple_vocab["a"]]
        mask = get_valid_token_mask(fsm, state_id=terminal, vocab_size=26)
        assert not mask.any()

    def test_invalid_state_returns_empty_mask(self, simple_vocab):
        """Unknown state ID should return all-False mask."""
        fsm = build_fsm_from_choices(["a"], simple_vocab)
        mask = get_valid_token_mask(fsm, state_id=999, vocab_size=26)
        assert not mask.any()

    def test_mask_shape(self, simple_vocab):
        """Mask shape should match vocab_size."""
        fsm = build_fsm_from_choices(["a"], simple_vocab)
        mask = get_valid_token_mask(fsm, state_id=0, vocab_size=100)
        assert mask.shape == (100,)


class TestConstrainedSample:
    def test_samples_valid_token(self):
        """Should only sample from valid positions."""
        logits = torch.randn(10)
        valid_mask = torch.zeros(10, dtype=torch.bool)
        valid_mask[3] = True  # only token 3 is valid
        for _ in range(10):
            token = constrained_sample(logits, valid_mask, temperature=1.0)
            assert token == 3

    def test_no_valid_returns_neg_one(self):
        """When no tokens are valid, should return -1."""
        logits = torch.randn(10)
        valid_mask = torch.zeros(10, dtype=torch.bool)
        assert constrained_sample(logits, valid_mask) == -1

    def test_full_decoding_produces_valid_choice(self, simple_vocab):
        """Full constrained decoding loop should produce one of the valid choices."""
        choices = ["yes", "no", "maybe"]
        fsm = build_fsm_from_choices(choices, simple_vocab)
        vocab_size = 26

        # Run constrained decoding
        torch.manual_seed(42)
        state = 0
        generated = []
        for _ in range(10):  # max steps
            mask = get_valid_token_mask(fsm, state, vocab_size)
            if not mask.any():
                break
            logits = torch.randn(vocab_size)
            token = constrained_sample(logits, mask, temperature=0.5)
            generated.append(token)
            state = fsm[state][token]

        # Reconstruct string
        id_to_char = {v: k for k, v in simple_vocab.items()}
        result = "".join(id_to_char[t] for t in generated)
        assert result in choices

    def test_low_temperature_picks_highest_logit(self):
        """Very low temperature should pick the highest-logit valid token."""
        logits = torch.tensor([1.0, 5.0, 3.0, 10.0, 2.0])
        valid_mask = torch.tensor([True, True, True, False, True])
        # Token 3 has highest logit but is invalid; token 1 (5.0) should win
        for _ in range(10):
            token = constrained_sample(logits, valid_mask, temperature=0.01)
            assert token == 1
