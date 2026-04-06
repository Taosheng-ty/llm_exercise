"""Tests for Exercise 3: Reward Shaping"""

import pytest
from .solution import compute_length_penalty, compute_format_bonus, shape_reward


class TestLengthPenalty:
    def test_short_response_no_penalty(self):
        assert compute_length_penalty("short", 100) == 0.0

    def test_exact_length_no_penalty(self):
        assert compute_length_penalty("x" * 100, 100) == 0.0

    def test_double_length(self):
        result = compute_length_penalty("x" * 200, 100, alpha=0.5)
        assert abs(result - (-0.5)) < 1e-9

    def test_alpha_scaling(self):
        result = compute_length_penalty("x" * 200, 100, alpha=1.0)
        assert abs(result - (-1.0)) < 1e-9

    def test_slightly_over(self):
        result = compute_length_penalty("x" * 110, 100, alpha=0.5)
        assert abs(result - (-0.05)) < 1e-9


class TestFormatBonus:
    def test_step_marker(self):
        assert compute_format_bonus("Step 1: do something") == 0.1

    def test_therefore_marker(self):
        assert compute_format_bonus("Therefore the answer is 5") == 0.1

    def test_first_comma_marker(self):
        assert compute_format_bonus("First, we compute the sum") == 0.1

    def test_thus_comma_marker(self):
        assert compute_format_bonus("Thus, we get 42") == 0.1

    def test_in_conclusion_marker(self):
        assert compute_format_bonus("In conclusion the result is 7") == 0.1

    def test_no_markers(self):
        assert compute_format_bonus("The answer is 5.") == 0.0

    def test_case_insensitive(self):
        assert compute_format_bonus("STEP 1: start") == 0.1

    def test_custom_beta(self):
        assert compute_format_bonus("Step 1: go", beta=0.5) == 0.5


class TestShapeReward:
    def test_correct_short_formatted(self):
        # raw=1.0, short response with marker -> 1.0 + 0.0 + 0.1 = 1.1
        result = shape_reward(1.0, "Step 1: answer is 5", target_length=500)
        assert abs(result - 1.1) < 1e-9

    def test_correct_long_no_format(self):
        # raw=1.0, 1000 chars, no marker -> 1.0 + penalty + 0.0
        response = "x" * 1000
        result = shape_reward(1.0, response, target_length=500, alpha=0.5)
        expected = 1.0 + (-0.5 * 500 / 500)
        assert abs(result - expected) < 1e-9

    def test_incorrect_short_no_format(self):
        # raw=0.0, short, no marker -> 0.0
        result = shape_reward(0.0, "wrong", target_length=500)
        assert abs(result - 0.0) < 1e-9

    def test_incorrect_with_format_bonus(self):
        # raw=0.0, short, has marker -> 0.0 + 0.0 + 0.1 = 0.1
        result = shape_reward(0.0, "Step 1: wrong answer", target_length=500)
        assert abs(result - 0.1) < 1e-9
