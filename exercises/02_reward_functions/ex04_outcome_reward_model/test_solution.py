"""Tests for Exercise 4: Outcome Reward Model"""

import pytest
from .solution import (
    extract_from_boxed,
    extract_from_answer_phrase,
    extract_from_last_line,
    normalize_for_comparison,
    compute_outcome_reward,
)


class TestExtractFromBoxed:
    def test_simple_boxed(self):
        assert extract_from_boxed("The answer is \\boxed{42}") == "42"

    def test_nested_braces(self):
        assert extract_from_boxed("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"

    def test_no_boxed(self):
        assert extract_from_boxed("No boxed here") is None

    def test_last_boxed_used(self):
        resp = "\\boxed{wrong} and then \\boxed{42}"
        assert extract_from_boxed(resp) == "42"

    def test_deeply_nested(self):
        assert extract_from_boxed("\\boxed{a{b{c}}}") == "a{b{c}}"


class TestExtractFromAnswerPhrase:
    def test_basic_phrase(self):
        assert extract_from_answer_phrase("The answer is 42.") == "42"

    def test_case_insensitive(self):
        assert extract_from_answer_phrase("the answer is 7") == "7"

    def test_no_match(self):
        assert extract_from_answer_phrase("I computed 42") is None

    def test_last_occurrence(self):
        resp = "The answer is 3. Wait, the answer is 7."
        assert extract_from_answer_phrase(resp) == "7"

    def test_with_newline(self):
        resp = "The answer is 42\nSome extra text"
        assert extract_from_answer_phrase(resp) == "42"


class TestExtractFromLastLine:
    def test_single_line(self):
        assert extract_from_last_line("42") == "42"

    def test_multi_line(self):
        assert extract_from_last_line("Step 1\nStep 2\n42") == "42"

    def test_trailing_blank_lines(self):
        assert extract_from_last_line("hello\n42\n  \n") == "42"

    def test_empty_string(self):
        assert extract_from_last_line("") is None


class TestNormalizeForComparison:
    def test_lowercase_strip(self):
        assert normalize_for_comparison("  Hello  ") == "hello"

    def test_remove_dollar(self):
        assert normalize_for_comparison("$42$") == "42"

    def test_remove_trailing_period(self):
        assert normalize_for_comparison("42.") == "42"


class TestComputeOutcomeReward:
    def test_boxed_correct(self):
        assert compute_outcome_reward("42", "Therefore \\boxed{42}") == 1.0

    def test_boxed_incorrect(self):
        assert compute_outcome_reward("42", "Therefore \\boxed{43}") == 0.0

    def test_answer_phrase_correct(self):
        assert compute_outcome_reward("7", "Thinking... The answer is 7.") == 1.0

    def test_last_line_fallback(self):
        assert compute_outcome_reward("42", "Some reasoning\n42") == 1.0

    def test_no_answer_found(self):
        assert compute_outcome_reward("42", "") == 0.0

    def test_boxed_takes_priority(self):
        # boxed says 5, answer phrase says 10 -- boxed wins
        resp = "The answer is 10. But actually \\boxed{5}"
        assert compute_outcome_reward("5", resp) == 1.0
        assert compute_outcome_reward("10", resp) == 0.0

    def test_case_insensitive_comparison(self):
        assert compute_outcome_reward("YES", "The answer is yes.") == 1.0
