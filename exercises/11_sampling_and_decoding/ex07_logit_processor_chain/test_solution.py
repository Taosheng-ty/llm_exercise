"""Tests for Exercise 07: Logit Processor Chain"""

import importlib.util
import os

import numpy as np
import pytest

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

TemperatureProcessor = _mod.TemperatureProcessor
TopKProcessor = _mod.TopKProcessor
TopPProcessor = _mod.TopPProcessor
RepetitionPenaltyProcessor = _mod.RepetitionPenaltyProcessor
LogitProcessorChain = _mod.LogitProcessorChain


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


class TestTemperatureProcessor:
    def test_temp_one_unchanged(self):
        logits = np.array([1.0, 2.0, 3.0])
        result = TemperatureProcessor(1.0)(np.array([]), logits)
        np.testing.assert_allclose(result, logits)

    def test_temp_half(self):
        logits = np.array([1.0, 2.0, 3.0])
        result = TemperatureProcessor(0.5)(np.array([]), logits)
        np.testing.assert_allclose(result, logits * 2)

    def test_temp_zero_greedy(self):
        logits = np.array([1.0, 5.0, 3.0])
        result = TemperatureProcessor(0.0)(np.array([]), logits)
        probs = _softmax(result)
        assert np.argmax(probs) == 1
        assert probs[1] == pytest.approx(1.0, abs=1e-5)


class TestTopKProcessor:
    def test_k_one(self):
        logits = np.array([1.0, 5.0, 3.0, 2.0])
        result = TopKProcessor(1)(np.array([]), logits)
        finite = np.isfinite(result)
        assert finite.sum() == 1
        assert finite[1]

    def test_k_all(self):
        logits = np.array([1.0, 2.0, 3.0])
        result = TopKProcessor(5)(np.array([]), logits)
        np.testing.assert_allclose(result, logits)

    def test_k_two(self):
        logits = np.array([1.0, 5.0, 3.0, 2.0])
        result = TopKProcessor(2)(np.array([]), logits)
        assert np.isfinite(result[1])  # 5.0
        assert np.isfinite(result[2])  # 3.0
        assert not np.isfinite(result[0])
        assert not np.isfinite(result[3])


class TestTopPProcessor:
    def test_p_one_keeps_all(self):
        logits = np.array([1.0, 2.0, 3.0])
        result = TopPProcessor(1.0)(np.array([]), logits)
        assert np.all(np.isfinite(result))

    def test_small_p_greedy(self):
        logits = np.array([1.0, 10.0, 2.0])
        result = TopPProcessor(0.01)(np.array([]), logits)
        probs = _softmax(result)
        assert probs[1] == pytest.approx(1.0, abs=1e-3)

    def test_always_keeps_one(self):
        logits = np.array([5.0, 5.0, 5.0])
        result = TopPProcessor(0.001)(np.array([]), logits)
        assert np.isfinite(result).sum() >= 1


class TestRepetitionPenaltyProcessor:
    def test_no_penalty(self):
        logits = np.array([1.0, 2.0, 3.0])
        result = RepetitionPenaltyProcessor(1.0)(np.array([0, 1]), logits)
        np.testing.assert_allclose(result, logits)

    def test_positive_divided(self):
        logits = np.array([4.0, 2.0, 1.0])
        result = RepetitionPenaltyProcessor(2.0)(np.array([0]), logits)
        assert result[0] == pytest.approx(2.0)
        assert result[1] == pytest.approx(2.0)

    def test_negative_multiplied(self):
        logits = np.array([-2.0, 3.0, 1.0])
        result = RepetitionPenaltyProcessor(2.0)(np.array([0]), logits)
        assert result[0] == pytest.approx(-4.0)

    def test_empty_ids(self):
        logits = np.array([1.0, 2.0])
        result = RepetitionPenaltyProcessor(2.0)(np.array([]), logits)
        np.testing.assert_allclose(result, logits)


class TestLogitProcessorChain:
    def test_chain_applies_in_order(self):
        """Temperature then top-k should work."""
        logits = np.array([1.0, 2.0, 3.0, 4.0])
        chain = LogitProcessorChain([
            TemperatureProcessor(0.5),
            TopKProcessor(2),
        ])
        result = chain(np.array([]), logits)
        # After temp=0.5: [2, 4, 6, 8]
        # After top-k=2: [-inf, -inf, 6, 8]
        assert np.isfinite(result).sum() == 2
        assert result[3] == pytest.approx(8.0)
        assert result[2] == pytest.approx(6.0)

    def test_empty_chain(self):
        """Empty chain should return logits unchanged."""
        logits = np.array([1.0, 2.0, 3.0])
        chain = LogitProcessorChain([])
        result = chain(np.array([]), logits)
        np.testing.assert_allclose(result, logits)

    def test_full_pipeline(self):
        """Full pipeline: repetition penalty -> temperature -> top-k -> top-p."""
        logits = np.array([4.0, 2.0, 1.0, 0.5, -1.0])
        chain = LogitProcessorChain([
            RepetitionPenaltyProcessor(1.5),
            TemperatureProcessor(0.8),
            TopKProcessor(3),
            TopPProcessor(0.95),
        ])
        result = chain(np.array([0, 2]), logits)
        # Should produce valid filtered logits
        assert np.isfinite(result).sum() >= 1
        assert result.shape == logits.shape

    def test_chain_does_not_modify_input(self):
        """Chain should not modify the original logits array."""
        logits = np.array([1.0, 2.0, 3.0])
        original = logits.copy()
        chain = LogitProcessorChain([TemperatureProcessor(0.5)])
        chain(np.array([]), logits)
        np.testing.assert_allclose(logits, original)
