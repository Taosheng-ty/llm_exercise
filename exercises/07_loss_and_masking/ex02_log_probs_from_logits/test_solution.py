"""Tests for Exercise 02: Extract Per-Token Log Probabilities from Logits."""
import numpy as np
import pytest

from .solution import log_probs_from_logits


class TestLogProbsFromLogits:
    def test_output_shape(self):
        """Output shape should be (batch, seq_len)."""
        logits = np.random.randn(3, 5, 10)
        token_ids = np.random.randint(0, 10, (3, 5))
        result = log_probs_from_logits(logits, token_ids)
        assert result.shape == (3, 5)

    def test_log_probs_are_negative(self):
        """Log probabilities should always be <= 0."""
        logits = np.random.randn(2, 4, 8)
        token_ids = np.random.randint(0, 8, (2, 4))
        result = log_probs_from_logits(logits, token_ids)
        assert np.all(result <= 1e-10), "Log probs should be non-positive"

    def test_uniform_distribution(self):
        """With uniform logits, all log probs should equal -log(V)."""
        vocab_size = 20
        logits = np.zeros((1, 3, vocab_size))
        token_ids = np.array([[0, 10, 19]])
        result = log_probs_from_logits(logits, token_ids)
        expected = -np.log(vocab_size)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_peaked_distribution(self):
        """When one logit dominates, its log prob should be near 0."""
        logits = np.full((1, 1, 5), -100.0)
        logits[0, 0, 3] = 100.0
        token_ids = np.array([[3]])
        result = log_probs_from_logits(logits, token_ids)
        assert result[0, 0] > -1e-5, f"Expected near 0, got {result[0, 0]}"

    def test_peaked_distribution_wrong_token(self):
        """Picking the wrong token from a peaked distribution should give
        a very negative log prob."""
        logits = np.full((1, 1, 5), -100.0)
        logits[0, 0, 3] = 100.0
        token_ids = np.array([[0]])  # wrong token
        result = log_probs_from_logits(logits, token_ids)
        assert result[0, 0] < -50.0, f"Expected very negative, got {result[0, 0]}"

    def test_numerical_stability(self):
        """Should handle very large logit values without overflow."""
        logits = np.array([[[1000.0, 1001.0, 999.0]]])
        token_ids = np.array([[1]])
        result = log_probs_from_logits(logits, token_ids)
        assert np.all(np.isfinite(result))
        # Token 1 has the highest logit, so its log prob should be close to 0
        assert result[0, 0] > -2.0

    def test_consistency_with_manual_computation(self):
        """Compare against a direct manual computation."""
        logits = np.array([[[2.0, 1.0, 0.5]]])
        token_ids = np.array([[0]])
        result = log_probs_from_logits(logits, token_ids)

        # Manual: log(exp(2) / (exp(2) + exp(1) + exp(0.5)))
        exp_vals = np.exp(logits[0, 0])
        expected = np.log(exp_vals[0] / exp_vals.sum())
        np.testing.assert_allclose(result[0, 0], expected, atol=1e-6)
