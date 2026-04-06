"""Tests for Exercise 01: Cross-Entropy Loss for Language Modeling."""
import numpy as np
import pytest

from .solution import cross_entropy_loss, log_softmax


class TestLogSoftmax:
    def test_sums_to_zero_in_log_space(self):
        """exp(log_softmax) should sum to 1 along last axis."""
        logits = np.random.randn(2, 5, 10)
        result = log_softmax(logits)
        probs = np.exp(result)
        np.testing.assert_allclose(probs.sum(axis=-1), 1.0, atol=1e-6)

    def test_numerical_stability_large_logits(self):
        """Should not overflow with very large logit values."""
        logits = np.array([[1000.0, 1001.0, 999.0]])
        result = log_softmax(logits)
        assert np.all(np.isfinite(result)), "log_softmax produced non-finite values"
        probs = np.exp(result)
        np.testing.assert_allclose(probs.sum(axis=-1), 1.0, atol=1e-6)

    def test_uniform_distribution(self):
        """Equal logits should produce uniform log-softmax."""
        logits = np.ones((1, 4))
        result = log_softmax(logits)
        expected = np.full((1, 4), -np.log(4))
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_shape_preserved(self):
        """Output shape should match input shape."""
        logits = np.random.randn(3, 7, 50)
        result = log_softmax(logits)
        assert result.shape == logits.shape


class TestCrossEntropyLoss:
    def test_perfect_prediction(self):
        """When logits strongly favor the correct token, loss should be near 0."""
        logits = np.full((1, 2, 5), -100.0)
        targets = np.array([[0, 2]])
        logits[0, 0, 0] = 100.0
        logits[0, 1, 2] = 100.0
        loss_mask = np.ones((1, 2))
        loss = cross_entropy_loss(logits, targets, loss_mask)
        assert loss < 1e-5, f"Expected near-zero loss, got {loss}"

    def test_uniform_logits(self):
        """With uniform logits over V classes, loss should be log(V)."""
        vocab_size = 10
        logits = np.zeros((1, 3, vocab_size))
        targets = np.array([[0, 5, 9]])
        loss_mask = np.ones((1, 3))
        loss = cross_entropy_loss(logits, targets, loss_mask)
        np.testing.assert_allclose(loss, np.log(vocab_size), atol=1e-6)

    def test_loss_mask_zeros_out_positions(self):
        """Masked positions should not contribute to the loss."""
        vocab_size = 5
        logits = np.zeros((1, 4, vocab_size))
        targets = np.array([[0, 1, 2, 3]])
        # Only compute loss on positions 2 and 3
        loss_mask = np.array([[0, 0, 1, 1]], dtype=float)
        loss = cross_entropy_loss(logits, targets, loss_mask)
        # Uniform logits -> loss = log(5) for unmasked positions
        np.testing.assert_allclose(loss, np.log(vocab_size), atol=1e-6)

    def test_empty_mask_returns_zero(self):
        """If all positions are masked, should return 0.0."""
        logits = np.random.randn(2, 5, 10)
        targets = np.zeros((2, 5), dtype=int)
        loss_mask = np.zeros((2, 5))
        loss = cross_entropy_loss(logits, targets, loss_mask)
        assert loss == 0.0

    def test_batch_consistency(self):
        """Loss computed on a batch should equal average of individual losses
        when each sample has the same number of unmasked tokens."""
        np.random.seed(42)
        vocab_size = 8
        logits = np.random.randn(2, 3, vocab_size)
        targets = np.random.randint(0, vocab_size, (2, 3))
        loss_mask = np.ones((2, 3))

        batch_loss = cross_entropy_loss(logits, targets, loss_mask)

        loss_0 = cross_entropy_loss(logits[0:1], targets[0:1], loss_mask[0:1])
        loss_1 = cross_entropy_loss(logits[1:2], targets[1:2], loss_mask[1:2])
        avg_loss = (loss_0 + loss_1) / 2.0

        np.testing.assert_allclose(batch_loss, avg_loss, atol=1e-6)
