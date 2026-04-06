"""Tests for Exercise 2: Token-level F1 Score"""

import pytest
from .solution import normalize_answer, f1_score


class TestNormalizeAnswer:
    def test_lowercase(self):
        assert normalize_answer("HELLO World") == "hello world"

    def test_remove_punctuation(self):
        assert normalize_answer("hello, world!") == "hello world"

    def test_remove_articles(self):
        assert normalize_answer("the cat and a dog") == "cat and dog"

    def test_whitespace_collapse(self):
        assert normalize_answer("  hello   world  ") == "hello world"

    def test_combined(self):
        assert normalize_answer("The Quick, Brown Fox!") == "quick brown fox"


class TestF1Score:
    def test_exact_match(self):
        f1, prec, rec = f1_score("cat sat", "cat sat")
        assert f1 == 1.0
        assert prec == 1.0
        assert rec == 1.0

    def test_partial_overlap(self):
        f1, prec, rec = f1_score("the cat sat", "a cat sat down")
        # After normalization: "cat sat" vs "cat sat down"
        # common=2, pred_len=2, gold_len=3
        assert prec == 1.0
        assert abs(rec - 2 / 3) < 1e-9
        assert abs(f1 - 0.8) < 1e-9

    def test_no_overlap(self):
        f1, prec, rec = f1_score("hello", "world")
        assert (f1, prec, rec) == (0.0, 0.0, 0.0)

    def test_none_prediction(self):
        assert f1_score(None, "something") == (0.0, 0.0, 0.0)

    def test_special_token_mismatch(self):
        assert f1_score("yes", "no") == (0.0, 0.0, 0.0)

    def test_special_token_match(self):
        f1, prec, rec = f1_score("yes", "yes")
        assert f1 == 1.0

    def test_articles_dont_count(self):
        # "the answer" -> "answer", "an answer" -> "answer"
        f1, prec, rec = f1_score("the answer", "an answer")
        assert f1 == 1.0
