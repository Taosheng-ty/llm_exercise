"""Tests for Exercise 05: Beam Search Decoding"""

import importlib.util
import math
import os

import pytest
import torch

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
beam_search = _mod.beam_search


def make_deterministic_log_prob_fn(vocab_size, preferred_sequence):
    """Create a log_prob_fn that strongly prefers a specific token sequence.

    The preferred token at each position gets log_prob = -0.1,
    all others get log_prob = -10.0.
    """
    def log_prob_fn(token_ids):
        step = len(token_ids) - 1  # subtract BOS
        log_probs = torch.full((vocab_size,), -10.0)
        if step < len(preferred_sequence):
            log_probs[preferred_sequence[step]] = -0.1
        else:
            # After preferred sequence, prefer EOS (token 1)
            log_probs[1] = -0.1
        return log_probs
    return log_prob_fn


class TestBeamSearch:
    def test_greedy_with_beam_1(self):
        """beam_width=1 should behave like greedy search."""
        # Preferred sequence: [2, 3, EOS(1)]
        fn = make_deterministic_log_prob_fn(vocab_size=5, preferred_sequence=[2, 3, 1])
        results = beam_search(fn, beam_width=1, max_length=5, eos_token_id=1)
        assert len(results) >= 1
        tokens, score = results[0]
        # Should follow preferred path: BOS(0), 2, 3, EOS(1)
        assert tokens == [0, 2, 3, 1]

    def test_multiple_beams(self):
        """beam_width > 1 should return multiple sequences."""
        fn = make_deterministic_log_prob_fn(vocab_size=4, preferred_sequence=[2, 1])
        results = beam_search(fn, beam_width=3, max_length=5, eos_token_id=1)
        assert len(results) >= 1
        # Best result should follow preferred path
        assert results[0][0] == [0, 2, 1]

    def test_eos_stops_beam(self):
        """Beams that hit EOS should stop expanding."""
        fn = make_deterministic_log_prob_fn(vocab_size=4, preferred_sequence=[1])
        results = beam_search(fn, beam_width=2, max_length=10, eos_token_id=1)
        # Top beam should be short: [BOS, EOS]
        assert results[0][0] == [0, 1]

    def test_max_length_enforced(self):
        """No sequence should exceed max_length."""
        # Never generate EOS
        def fn(token_ids):
            log_probs = torch.full((4,), -1.0)
            log_probs[1] = -100.0  # EOS very unlikely
            log_probs[2] = -0.5    # prefer token 2
            return log_probs

        results = beam_search(fn, beam_width=2, max_length=4, eos_token_id=1)
        for tokens, _ in results:
            assert len(tokens) <= 4

    def test_length_penalty(self):
        """Length penalty should normalize scores by length."""
        fn = make_deterministic_log_prob_fn(vocab_size=4, preferred_sequence=[2, 3, 1])
        # Without length penalty
        results_no_penalty = beam_search(
            fn, beam_width=2, max_length=6, eos_token_id=1, length_penalty=0.0
        )
        # With length penalty
        results_with_penalty = beam_search(
            fn, beam_width=2, max_length=6, eos_token_id=1, length_penalty=1.0
        )
        # Scores should differ due to normalization
        if len(results_no_penalty) > 0 and len(results_with_penalty) > 0:
            # Same best sequence, but different score
            assert results_no_penalty[0][0] == results_with_penalty[0][0]
            # With penalty, score = raw_score / length
            raw_score = results_no_penalty[0][1]
            length = len(results_with_penalty[0][0])
            expected = raw_score / length
            assert results_with_penalty[0][1] == pytest.approx(expected, abs=1e-5)

    def test_scores_are_negative(self):
        """Log probability scores should be non-positive."""
        fn = make_deterministic_log_prob_fn(vocab_size=4, preferred_sequence=[2, 1])
        results = beam_search(fn, beam_width=2, max_length=5, eos_token_id=1)
        for _, score in results:
            assert score <= 0.0

    def test_results_sorted_by_score(self):
        """Results should be sorted by normalized score, best first."""
        fn = make_deterministic_log_prob_fn(vocab_size=5, preferred_sequence=[2, 3, 1])
        results = beam_search(fn, beam_width=3, max_length=6, eos_token_id=1)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_bos_token_present(self):
        """All sequences should start with BOS token (0)."""
        fn = make_deterministic_log_prob_fn(vocab_size=4, preferred_sequence=[2, 1])
        results = beam_search(fn, beam_width=2, max_length=5, eos_token_id=1)
        for tokens, _ in results:
            assert tokens[0] == 0
