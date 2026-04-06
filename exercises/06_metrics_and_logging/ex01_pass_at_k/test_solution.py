"""Tests for Exercise 01: Pass@k Metric Estimation."""

import math
import importlib
import os
import sys

import numpy as np
import pytest

# Import solution from the same directory as this test file
_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("solution_ex01", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
estimate_pass_at_k = _mod.estimate_pass_at_k
estimate_pass_at_k_batch = _mod.estimate_pass_at_k_batch
compute_pass_rates = _mod.compute_pass_rates


class TestEstimatePassAtK:
    def test_all_correct(self):
        """If all samples are correct, pass@k should be 1.0."""
        assert estimate_pass_at_k(10, 10, 1) == 1.0
        assert estimate_pass_at_k(10, 10, 5) == 1.0
        assert estimate_pass_at_k(10, 10, 10) == 1.0

    def test_none_correct(self):
        """If no samples are correct, pass@k should be 0.0."""
        assert estimate_pass_at_k(10, 0, 1) == 0.0
        assert estimate_pass_at_k(10, 0, 5) == 0.0
        assert estimate_pass_at_k(10, 0, 10) == 0.0

    def test_pass_at_1_simple(self):
        """pass@1 with c correct out of n is c/n."""
        # pass@1 = 1 - C(n-c,1)/C(n,1) = 1 - (n-c)/n = c/n
        assert estimate_pass_at_k(10, 3, 1) == pytest.approx(0.3)
        assert estimate_pass_at_k(4, 2, 1) == pytest.approx(0.5)

    def test_pass_at_k_equals_n(self):
        """When k == n, if c > 0 then pass@k = 1.0."""
        assert estimate_pass_at_k(5, 1, 5) == 1.0
        assert estimate_pass_at_k(5, 3, 5) == 1.0

    def test_k_greater_than_n(self):
        """When k > n, return 1.0 if c > 0, else 0.0."""
        assert estimate_pass_at_k(5, 3, 10) == 1.0
        assert estimate_pass_at_k(5, 0, 10) == 0.0

    def test_known_value(self):
        """Test a known computation: n=10, c=3, k=2."""
        # pass@2 = 1 - C(7,2)/C(10,2) = 1 - 21/45 = 1 - 7/15 = 8/15
        expected = 1.0 - math.comb(7, 2) / math.comb(10, 2)
        assert estimate_pass_at_k(10, 3, 2) == pytest.approx(expected)


class TestEstimatePassAtKBatch:
    def test_multiple_problems(self):
        """Batch estimation for multiple problems."""
        result = estimate_pass_at_k_batch([10, 10, 10], [5, 0, 10], 1)
        assert len(result) == 3
        assert result[0] == pytest.approx(0.5)
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(1.0)

    def test_returns_numpy_array(self):
        result = estimate_pass_at_k_batch([4], [2], 1)
        assert isinstance(result, np.ndarray)


class TestComputePassRates:
    def test_group_size_1_returns_empty(self):
        """group_size < 2 should return empty dict."""
        assert compute_pass_rates([1.0, 0.0], 1) == {}

    def test_all_correct(self):
        """All correct rewards => all pass@k = 1.0."""
        rewards = [1.0] * 8  # 2 problems, 4 samples each
        result = compute_pass_rates(rewards, group_size=4)
        assert "pass@1" in result
        assert "pass@2" in result
        assert "pass@4" in result
        for v in result.values():
            assert v == pytest.approx(1.0)

    def test_all_incorrect(self):
        """All incorrect rewards => all pass@k = 0.0."""
        rewards = [0.0] * 8
        result = compute_pass_rates(rewards, group_size=4)
        for v in result.values():
            assert v == pytest.approx(0.0)

    def test_keys_are_powers_of_2(self):
        """Keys should be pass@1, pass@2, ..., pass@group_size (powers of 2)."""
        rewards = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]
        result = compute_pass_rates(rewards, group_size=4)
        assert set(result.keys()) == {"pass@1", "pass@2", "pass@4"}

    def test_mixed_rewards(self):
        """Test with mixed rewards: 2 problems, group_size=4."""
        # Problem 1: [1,0,1,0] -> c=2, Problem 2: [0,0,0,1] -> c=1
        rewards = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        result = compute_pass_rates(rewards, group_size=4)
        # pass@1 for problem1 = 2/4 = 0.5, problem2 = 1/4 = 0.25 => avg = 0.375
        assert result["pass@1"] == pytest.approx(0.375)
        # pass@4 = both have c>0, so both 1.0 => avg = 1.0
        assert result["pass@4"] == pytest.approx(1.0)

    def test_group_size_8(self):
        """group_size=8 should produce pass@1, pass@2, pass@4, pass@8."""
        rewards = [1.0, 0.0] * 4  # 1 problem, 8 samples, 4 correct
        result = compute_pass_rates(rewards, group_size=8)
        assert set(result.keys()) == {"pass@1", "pass@2", "pass@4", "pass@8"}
        assert result["pass@8"] == pytest.approx(1.0)
