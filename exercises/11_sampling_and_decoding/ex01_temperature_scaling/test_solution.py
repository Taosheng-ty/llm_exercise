"""Tests for Exercise 01: Temperature Scaling"""

import importlib.util
import os

import pytest
import torch

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
temperature_scale = _mod.temperature_scale


class TestTemperatureScaling:
    def test_temperature_one_unchanged(self):
        """Temperature=1 should leave logits unchanged."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = temperature_scale(logits, temperature=1.0)
        torch.testing.assert_close(result, logits)

    def test_temperature_half_doubles(self):
        """Temperature=0.5 should double the logits."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = temperature_scale(logits, temperature=0.5)
        torch.testing.assert_close(result, logits * 2.0)

    def test_temperature_two_halves(self):
        """Temperature=2 should halve the logits."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = temperature_scale(logits, temperature=2.0)
        torch.testing.assert_close(result, logits / 2.0)

    def test_temperature_zero_greedy(self):
        """Temperature=0 should produce argmax behavior."""
        logits = torch.tensor([[1.0, 5.0, 3.0]])
        result = temperature_scale(logits, temperature=0.0)
        probs = torch.softmax(result, dim=-1)
        # All probability mass on index 1 (the argmax)
        assert probs[0, 1].item() == pytest.approx(1.0, abs=1e-5)
        assert probs[0, 0].item() == pytest.approx(0.0, abs=1e-5)
        assert probs[0, 2].item() == pytest.approx(0.0, abs=1e-5)

    def test_temperature_zero_batch(self):
        """Temperature=0 with batched input."""
        logits = torch.tensor([[1.0, 5.0, 3.0], [7.0, 2.0, 4.0]])
        result = temperature_scale(logits, temperature=0.0)
        probs = torch.softmax(result, dim=-1)
        assert probs[0].argmax().item() == 1
        assert probs[1].argmax().item() == 0

    def test_high_temperature_flattens(self):
        """Very high temperature should flatten the distribution toward uniform."""
        logits = torch.tensor([[1.0, 100.0, 1.0]])
        result = temperature_scale(logits, temperature=1e6)
        probs = torch.softmax(result, dim=-1)
        # Should be nearly uniform
        assert probs[0, 0].item() == pytest.approx(1.0 / 3, abs=0.01)
        assert probs[0, 1].item() == pytest.approx(1.0 / 3, abs=0.01)

    def test_output_shape(self):
        """Output shape should match input shape."""
        logits = torch.randn(4, 50)
        result = temperature_scale(logits, temperature=0.8)
        assert result.shape == logits.shape
