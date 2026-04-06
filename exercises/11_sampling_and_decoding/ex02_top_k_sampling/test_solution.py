"""Tests for Exercise 02: Top-K Sampling"""

import importlib.util
import os

import pytest
import torch

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
top_k_filter = _mod.top_k_filter


class TestTopKSampling:
    def test_k_equals_vocab_size_no_filtering(self):
        """k = vocab_size should leave logits unchanged."""
        logits = torch.tensor([[3.0, 1.0, 2.0, 5.0, 4.0]])
        result = top_k_filter(logits, k=5)
        torch.testing.assert_close(result, logits)

    def test_k_greater_than_vocab_no_filtering(self):
        """k > vocab_size should leave logits unchanged."""
        logits = torch.tensor([[3.0, 1.0, 2.0]])
        result = top_k_filter(logits, k=10)
        torch.testing.assert_close(result, logits)

    def test_k_one_greedy(self):
        """k=1 should keep only the maximum logit."""
        logits = torch.tensor([[1.0, 5.0, 3.0, 2.0]])
        result = top_k_filter(logits, k=1)
        probs = torch.softmax(result, dim=-1)
        assert probs[0, 1].item() == pytest.approx(1.0, abs=1e-5)

    def test_k_two_keeps_top_two(self):
        """k=2 should keep exactly the top 2 logits."""
        logits = torch.tensor([[1.0, 5.0, 3.0, 2.0]])
        result = top_k_filter(logits, k=2)
        # Top 2 are indices 1 (5.0) and 2 (3.0)
        assert result[0, 1].item() == 5.0
        assert result[0, 2].item() == 3.0
        assert result[0, 0].item() == float("-inf")
        assert result[0, 3].item() == float("-inf")

    def test_batch_processing(self):
        """Should handle batched inputs correctly."""
        logits = torch.tensor([
            [1.0, 5.0, 3.0],
            [7.0, 2.0, 4.0],
        ])
        result = top_k_filter(logits, k=2)
        # Row 0: top-2 are indices 1,2
        assert result[0, 0].item() == float("-inf")
        assert result[0, 1].item() == 5.0
        assert result[0, 2].item() == 3.0
        # Row 1: top-2 are indices 0,2
        assert result[1, 0].item() == 7.0
        assert result[1, 1].item() == float("-inf")
        assert result[1, 2].item() == 4.0

    def test_output_shape(self):
        """Output should preserve shape."""
        logits = torch.randn(3, 100)
        result = top_k_filter(logits, k=10)
        assert result.shape == logits.shape

    def test_non_top_k_are_neg_inf(self):
        """Non-top-k values should be exactly -inf."""
        logits = torch.tensor([[10.0, 1.0, 2.0, 3.0, 4.0]])
        result = top_k_filter(logits, k=2)
        finite_count = torch.isfinite(result).sum().item()
        assert finite_count == 2
