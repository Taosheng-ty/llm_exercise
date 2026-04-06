"""Tests for Exercise 04: Exact Match Metric."""

import importlib.util
import os

import pytest

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("solution_ex04", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
normalize_answer = _mod.normalize_answer
exact_match_single = _mod.exact_match_single
compute_exact_match = _mod.compute_exact_match


class TestNormalizeAnswer:
    def test_lowercase(self):
        assert normalize_answer("Hello World") == "hello world"

    def test_remove_punctuation(self):
        assert normalize_answer("hello, world!") == "hello world"

    def test_remove_articles(self):
        assert normalize_answer("the cat and a dog") == "cat and dog"

    def test_remove_an(self):
        assert normalize_answer("an apple") == "apple"

    def test_collapse_whitespace(self):
        assert normalize_answer("  lots   of   space  ") == "lots of space"

    def test_combined(self):
        assert normalize_answer("The Quick Brown Fox!") == "quick brown fox"

    def test_empty_string(self):
        assert normalize_answer("") == ""

    def test_only_articles_and_punctuation(self):
        assert normalize_answer("the, a, an.") == ""

    def test_possessive(self):
        # apostrophe is punctuation, so "dog's" -> "dogs"
        assert normalize_answer("A dog's life") == "dogs life"


class TestExactMatchSingle:
    def test_exact(self):
        assert exact_match_single("Paris", ["Paris"])

    def test_case_insensitive(self):
        assert exact_match_single("paris", ["Paris"])

    def test_with_article_difference(self):
        assert exact_match_single("The Eiffel Tower", ["Eiffel Tower"])

    def test_multiple_acceptable(self):
        assert exact_match_single("NYC", ["New York City", "NYC", "New York"])

    def test_no_match(self):
        assert not exact_match_single("London", ["Paris", "Berlin"])

    def test_punctuation_difference(self):
        assert exact_match_single("hello, world!", ["hello world"])

    def test_whitespace_difference(self):
        assert exact_match_single("  answer  ", ["answer"])


class TestComputeExactMatch:
    def test_perfect_score(self):
        preds = ["Paris", "42", "yes"]
        gts = [["Paris"], ["42"], ["yes"]]
        assert compute_exact_match(preds, gts) == 1.0

    def test_zero_score(self):
        preds = ["London", "43", "no"]
        gts = [["Paris"], ["42"], ["yes"]]
        assert compute_exact_match(preds, gts) == 0.0

    def test_partial_score(self):
        preds = ["Paris", "wrong", "yes"]
        gts = [["Paris"], ["42"], ["yes"]]
        score = compute_exact_match(preds, gts)
        assert abs(score - 2.0 / 3.0) < 1e-6

    def test_empty_predictions(self):
        assert compute_exact_match([], []) == 0.0

    def test_normalization_applied(self):
        preds = ["The Answer"]
        gts = [["answer"]]
        assert compute_exact_match(preds, gts) == 1.0

    def test_multiple_acceptable_answers(self):
        preds = ["NYC"]
        gts = [["New York City", "NYC"]]
        assert compute_exact_match(preds, gts) == 1.0
