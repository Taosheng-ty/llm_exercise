"""Tests for Exercise 04: Repetition Penalty"""

import importlib.util
import os

import pytest
import torch

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
apply_repetition_penalty = _mod.apply_repetition_penalty


class TestRepetitionPenalty:
    def test_no_penalty(self):
        """penalty=1.0 should leave logits unchanged."""
        logits = torch.tensor([[1.0, 2.0, 3.0, -1.0]])
        result = apply_repetition_penalty(logits, [[0, 1, 2]], penalty=1.0)
        torch.testing.assert_close(result, logits)

    def test_positive_logits_divided(self):
        """Positive logits for context tokens should be divided by penalty."""
        logits = torch.tensor([[4.0, 2.0, 1.0]])
        result = apply_repetition_penalty(logits, [[0]], penalty=2.0)
        assert result[0, 0].item() == pytest.approx(2.0)  # 4/2
        assert result[0, 1].item() == pytest.approx(2.0)  # unchanged
        assert result[0, 2].item() == pytest.approx(1.0)  # unchanged

    def test_negative_logits_multiplied(self):
        """Negative logits for context tokens should be multiplied by penalty."""
        logits = torch.tensor([[-2.0, 3.0, 1.0]])
        result = apply_repetition_penalty(logits, [[0]], penalty=2.0)
        assert result[0, 0].item() == pytest.approx(-4.0)  # -2*2
        assert result[0, 1].item() == pytest.approx(3.0)  # unchanged

    def test_zero_logit_unchanged(self):
        """Zero logit should remain zero (0/penalty=0, 0*penalty=0)."""
        logits = torch.tensor([[0.0, 3.0, -1.0]])
        result = apply_repetition_penalty(logits, [[0]], penalty=2.0)
        assert result[0, 0].item() == pytest.approx(0.0)

    def test_non_context_tokens_unchanged(self):
        """Tokens not in context should not be affected."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        result = apply_repetition_penalty(logits, [[0, 2]], penalty=2.0)
        assert result[0, 1].item() == pytest.approx(2.0)
        assert result[0, 3].item() == pytest.approx(4.0)

    def test_duplicate_token_ids(self):
        """Duplicate token IDs in context should work the same as unique."""
        logits = torch.tensor([[4.0, 2.0, 1.0]])
        result = apply_repetition_penalty(logits, [[0, 0, 0]], penalty=2.0)
        assert result[0, 0].item() == pytest.approx(2.0)  # 4/2, applied once

    def test_empty_context(self):
        """Empty context should leave logits unchanged."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = apply_repetition_penalty(logits, [[]], penalty=2.0)
        torch.testing.assert_close(result, logits)

    def test_batch_processing(self):
        """Should handle different contexts per batch element."""
        logits = torch.tensor([
            [4.0, 2.0, -1.0],
            [3.0, -2.0, 1.0],
        ])
        result = apply_repetition_penalty(logits, [[0], [1]], penalty=2.0)
        # Row 0: token 0 penalized (4/2=2), others unchanged
        assert result[0, 0].item() == pytest.approx(2.0)
        assert result[0, 1].item() == pytest.approx(2.0)
        # Row 1: token 1 penalized (-2*2=-4), others unchanged
        assert result[1, 1].item() == pytest.approx(-4.0)
        assert result[1, 0].item() == pytest.approx(3.0)

    def test_output_shape(self):
        """Output shape should match input."""
        logits = torch.randn(3, 50)
        result = apply_repetition_penalty(logits, [[], [1, 2], [0]], penalty=1.2)
        assert result.shape == logits.shape
