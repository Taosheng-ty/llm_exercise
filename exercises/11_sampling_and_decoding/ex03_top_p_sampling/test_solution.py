"""Tests for Exercise 03: Top-P (Nucleus) Sampling"""

import importlib.util
import os

import pytest
import torch

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
top_p_filter = _mod.top_p_filter


class TestTopPSampling:
    def test_p_one_no_filtering(self):
        """p=1.0 should keep all tokens (no filtering)."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        result = top_p_filter(logits, p=1.0)
        # All logits should remain finite
        assert torch.all(torch.isfinite(result)).item()

    def test_very_small_p_greedy(self):
        """Very small p should keep only the top token."""
        logits = torch.tensor([[1.0, 10.0, 2.0, 3.0]])
        result = top_p_filter(logits, p=0.01)
        probs = torch.softmax(result, dim=-1)
        # Only the top token (index 1) should have probability
        assert probs[0, 1].item() == pytest.approx(1.0, abs=1e-5)

    def test_always_keeps_at_least_one(self):
        """Even with p near 0, at least one token must survive."""
        logits = torch.tensor([[5.0, 5.0, 5.0]])
        result = top_p_filter(logits, p=0.001)
        finite_count = torch.isfinite(result).sum().item()
        assert finite_count >= 1

    def test_concentrated_distribution(self):
        """When one token dominates, only that token should survive at moderate p."""
        # Token 0 has prob ~0.997 after softmax of [10, 1, 1]
        logits = torch.tensor([[10.0, 1.0, 1.0]])
        result = top_p_filter(logits, p=0.9)
        # Only index 0 should survive
        assert torch.isfinite(result[0, 0]).item()
        assert result[0, 1].item() == float("-inf")
        assert result[0, 2].item() == float("-inf")

    def test_uniform_distribution_keeps_enough(self):
        """With uniform logits and p=0.5, roughly half should survive."""
        logits = torch.zeros(1, 10)  # uniform distribution
        result = top_p_filter(logits, p=0.5)
        finite_count = torch.isfinite(result).sum().item()
        # With 10 uniform tokens each at 0.1 prob, need 6 to exceed 0.5
        # (cumulative_probs - probs > p means we keep tokens where cum-prob <= p+prob)
        assert finite_count >= 5
        assert finite_count <= 7

    def test_batch_processing(self):
        """Should handle batched inputs."""
        logits = torch.tensor([
            [10.0, 1.0, 1.0],  # concentrated
            [1.0, 1.0, 1.0],   # uniform
        ])
        result = top_p_filter(logits, p=0.5)
        # Row 0: only top token should survive
        assert torch.isfinite(result[0, 0]).item()
        # Row 1: multiple tokens should survive
        assert torch.isfinite(result).sum(dim=-1)[1].item() >= 1

    def test_output_shape(self):
        """Output shape should match input shape."""
        logits = torch.randn(4, 50)
        result = top_p_filter(logits, p=0.9)
        assert result.shape == logits.shape

    def test_filtered_values_unchanged(self):
        """Surviving logit values should be unchanged from original."""
        logits = torch.tensor([[5.0, 3.0, 1.0, 0.5]])
        result = top_p_filter(logits, p=0.9)
        # For surviving tokens, values should match
        mask = torch.isfinite(result)
        torch.testing.assert_close(result[mask], logits[mask])
