"""Tests for Exercise 08: Stop Criteria"""

import importlib.util
import os

import numpy as np
import pytest

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

StopOnToken = _mod.StopOnToken
StopOnMaxLength = _mod.StopOnMaxLength
StopOnString = _mod.StopOnString
StopCriteriaChain = _mod.StopCriteriaChain


class TestStopOnToken:
    def test_stop_token_present(self):
        """Should return True when last token is the stop token."""
        stop = StopOnToken(stop_token_id=2)
        assert stop([1, 3, 2], "hello") is True

    def test_stop_token_absent(self):
        """Should return False when last token is not the stop token."""
        stop = StopOnToken(stop_token_id=2)
        assert stop([1, 3, 4], "hello") is False

    def test_empty_ids(self):
        """Should return False for empty generated_ids."""
        stop = StopOnToken(stop_token_id=2)
        assert stop([], "") is False

    def test_stop_token_not_last(self):
        """Should only check the last token."""
        stop = StopOnToken(stop_token_id=2)
        assert stop([2, 3, 4], "text") is False


class TestStopOnMaxLength:
    def test_under_max(self):
        assert StopOnMaxLength(5)([1, 2, 3], "abc") is False

    def test_at_max(self):
        assert StopOnMaxLength(3)([1, 2, 3], "abc") is True

    def test_over_max(self):
        assert StopOnMaxLength(2)([1, 2, 3], "abc") is True

    def test_zero_max(self):
        assert StopOnMaxLength(0)([], "") is True

    def test_one_token(self):
        assert StopOnMaxLength(1)([5], "x") is True


class TestStopOnString:
    def test_string_present(self):
        stop = StopOnString("end")
        assert stop([1, 2], "this is the end") is True

    def test_string_absent(self):
        stop = StopOnString("end")
        assert stop([1, 2], "this continues") is False

    def test_empty_text(self):
        stop = StopOnString("end")
        assert stop([], "") is False

    def test_partial_match(self):
        stop = StopOnString("hello world")
        assert stop([1], "hello worl") is False

    def test_substring_match(self):
        stop = StopOnString("<|endoftext|>")
        assert stop([1, 2, 3], "some text<|endoftext|>more") is True

    def test_empty_stop_string(self):
        """Empty string is always 'in' any string."""
        stop = StopOnString("")
        assert stop([1], "any text") is True


class TestStopCriteriaChain:
    def test_any_triggers(self):
        """Chain should stop if any criterion triggers."""
        chain = StopCriteriaChain([
            StopOnToken(2),
            StopOnMaxLength(10),
        ])
        # Token criterion triggers
        assert chain([1, 3, 2], "text") is True

    def test_none_triggers(self):
        """Chain should not stop if no criterion triggers."""
        chain = StopCriteriaChain([
            StopOnToken(2),
            StopOnMaxLength(10),
        ])
        assert chain([1, 3, 4], "text") is False

    def test_max_length_triggers(self):
        chain = StopCriteriaChain([
            StopOnToken(99),
            StopOnMaxLength(3),
        ])
        assert chain([1, 2, 3], "abc") is True

    def test_string_triggers(self):
        chain = StopCriteriaChain([
            StopOnToken(99),
            StopOnString("STOP"),
        ])
        assert chain([1, 2], "please STOP now") is True

    def test_empty_chain(self):
        """Empty chain should never trigger stop."""
        chain = StopCriteriaChain([])
        assert chain([1, 2, 3], "text") is False

    def test_all_criteria_combined(self):
        """Test with all four criteria types."""
        chain = StopCriteriaChain([
            StopOnToken(0),
            StopOnMaxLength(100),
            StopOnString("impossible_string_xyz"),
        ])
        # None should trigger
        assert chain([1, 2, 3], "normal text") is False
        # Token triggers
        assert chain([1, 2, 0], "normal text") is True
