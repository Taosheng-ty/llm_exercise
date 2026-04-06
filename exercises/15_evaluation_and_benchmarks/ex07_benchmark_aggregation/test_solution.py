"""Tests for Exercise 07: Benchmark Aggregation."""

import importlib.util
import os

import numpy as np
import pytest

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("solution_ex07", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
macro_average = _mod.macro_average
weighted_average = _mod.weighted_average
category_averages = _mod.category_averages
bootstrap_confidence_interval = _mod.bootstrap_confidence_interval


class TestMacroAverage:
    def test_basic(self):
        scores = {"gpqa": 0.4, "math": 0.6, "ifeval": 0.8}
        assert abs(macro_average(scores) - 0.6) < 1e-6

    def test_single_task(self):
        assert abs(macro_average({"task1": 0.75}) - 0.75) < 1e-6

    def test_empty(self):
        assert macro_average({}) == 0.0

    def test_all_same(self):
        scores = {"a": 0.5, "b": 0.5, "c": 0.5}
        assert abs(macro_average(scores) - 0.5) < 1e-6

    def test_zero_and_one(self):
        scores = {"a": 0.0, "b": 1.0}
        assert abs(macro_average(scores) - 0.5) < 1e-6


class TestWeightedAverage:
    def test_equal_weights(self):
        scores = {"a": 0.4, "b": 0.6}
        weights = {"a": 1.0, "b": 1.0}
        assert abs(weighted_average(scores, weights) - 0.5) < 1e-6

    def test_different_weights(self):
        scores = {"a": 0.0, "b": 1.0}
        weights = {"a": 1.0, "b": 3.0}
        # (0*1 + 1*3) / (1+3) = 0.75
        assert abs(weighted_average(scores, weights) - 0.75) < 1e-6

    def test_missing_weight_defaults_to_one(self):
        scores = {"a": 0.5, "b": 0.5}
        weights = {"a": 2.0}  # b gets weight 1.0
        # (0.5*2 + 0.5*1) / (2+1) = 1.5/3 = 0.5
        assert abs(weighted_average(scores, weights) - 0.5) < 1e-6

    def test_empty(self):
        assert weighted_average({}, {}) == 0.0

    def test_extra_weights_ignored(self):
        scores = {"a": 0.8}
        weights = {"a": 2.0, "b": 5.0}  # b not in scores
        assert abs(weighted_average(scores, weights) - 0.8) < 1e-6


class TestCategoryAverages:
    def test_basic_categories(self):
        scores = {"gpqa": 0.4, "math": 0.6, "ifeval": 0.8}
        categories = {"gpqa": "reasoning", "math": "reasoning", "ifeval": "instruction"}
        result = category_averages(scores, categories)
        assert abs(result["reasoning"] - 0.5) < 1e-6
        assert abs(result["instruction"] - 0.8) < 1e-6

    def test_uncategorized_goes_to_other(self):
        scores = {"a": 0.5, "b": 0.7}
        categories = {"a": "cat1"}
        result = category_averages(scores, categories)
        assert "cat1" in result
        assert "other" in result
        assert abs(result["other"] - 0.7) < 1e-6

    def test_single_category(self):
        scores = {"a": 0.3, "b": 0.7}
        categories = {"a": "all", "b": "all"}
        result = category_averages(scores, categories)
        assert abs(result["all"] - 0.5) < 1e-6

    def test_empty(self):
        assert category_averages({}, {}) == {}


class TestBootstrapConfidenceInterval:
    def test_constant_scores(self):
        """All same scores should give zero-width CI."""
        scores = [0.5] * 100
        result = bootstrap_confidence_interval(scores, n_bootstrap=500, seed=42)
        assert abs(result["mean"] - 0.5) < 1e-6
        assert abs(result["lower"] - 0.5) < 1e-6
        assert abs(result["upper"] - 0.5) < 1e-6
        assert result["std"] < 1e-6

    def test_binary_scores(self):
        """50/50 binary scores should have mean near 0.5."""
        scores = [1.0] * 50 + [0.0] * 50
        result = bootstrap_confidence_interval(scores, n_bootstrap=2000, seed=42)
        assert abs(result["mean"] - 0.5) < 0.05
        assert result["lower"] < result["mean"]
        assert result["upper"] > result["mean"]
        assert result["lower"] > 0.0
        assert result["upper"] < 1.0

    def test_ci_width_decreases_with_samples(self):
        """More samples should yield tighter CI."""
        scores_small = [0.5, 0.6, 0.4, 0.5, 0.6]
        scores_large = [0.5, 0.6, 0.4, 0.5, 0.6] * 20
        r_small = bootstrap_confidence_interval(scores_small, seed=42)
        r_large = bootstrap_confidence_interval(scores_large, seed=42)
        width_small = r_small["upper"] - r_small["lower"]
        width_large = r_large["upper"] - r_large["lower"]
        assert width_large < width_small

    def test_empty_scores(self):
        result = bootstrap_confidence_interval([])
        assert result["mean"] == 0.0
        assert result["lower"] == 0.0
        assert result["upper"] == 0.0
        assert result["std"] == 0.0

    def test_keys_present(self):
        result = bootstrap_confidence_interval([0.5, 0.6, 0.7])
        assert "mean" in result
        assert "lower" in result
        assert "upper" in result
        assert "std" in result

    def test_reproducibility(self):
        scores = [0.1, 0.9, 0.5, 0.3, 0.7]
        r1 = bootstrap_confidence_interval(scores, seed=123)
        r2 = bootstrap_confidence_interval(scores, seed=123)
        assert r1["mean"] == r2["mean"]
        assert r1["lower"] == r2["lower"]

    def test_confidence_level(self):
        """99% CI should be wider than 90% CI."""
        scores = list(np.random.RandomState(42).rand(100))
        r90 = bootstrap_confidence_interval(scores, confidence_level=0.90, seed=42)
        r99 = bootstrap_confidence_interval(scores, confidence_level=0.99, seed=42)
        width_90 = r90["upper"] - r90["lower"]
        width_99 = r99["upper"] - r99["lower"]
        assert width_99 > width_90
