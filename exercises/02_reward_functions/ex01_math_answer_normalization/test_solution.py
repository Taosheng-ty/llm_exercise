"""Tests for Exercise 1: Math Answer Normalization"""

import pytest
from .solution import normalize_math_answer, answers_are_equivalent


class TestNormalizeMathAnswer:
    def test_strip_dollar_signs(self):
        assert normalize_math_answer("$42$") == "42"

    def test_frac_conversion(self):
        assert normalize_math_answer("\\frac{1}{2}") == "1/2"

    def test_sqrt_conversion(self):
        assert normalize_math_answer("\\sqrt{3}") == "sqrt(3)"

    def test_variable_equals_value(self):
        assert normalize_math_answer("x = 7") == "7"

    def test_leading_decimal(self):
        assert normalize_math_answer(".5") == "0.5"

    def test_degree_removal(self):
        assert normalize_math_answer("90^{\\circ}") == "90"

    def test_percentage_removal(self):
        assert normalize_math_answer("50\\%") == "50"

    def test_tfrac_dfrac_replaced(self):
        result = normalize_math_answer("\\tfrac{3}{4}")
        assert result == "3/4"
        result2 = normalize_math_answer("\\dfrac{3}{4}")
        assert result2 == "3/4"

    def test_whitespace_and_newlines(self):
        assert normalize_math_answer("  42 \n") == "42"

    def test_unit_removal(self):
        assert normalize_math_answer("5\\text{ cm}") == "5"

    def test_lowercase(self):
        assert normalize_math_answer("X + Y") == "x+y"

    def test_left_right_removal(self):
        assert normalize_math_answer("\\left(1+2\\right)") == "(1+2)"


class TestAnswersAreEquivalent:
    def test_identical_after_normalization(self):
        assert answers_are_equivalent("42", " 42 ")

    def test_frac_vs_decimal(self):
        assert answers_are_equivalent("\\frac{1}{2}", "0.5")

    def test_different_frac_formats(self):
        assert answers_are_equivalent("\\tfrac{3}{4}", "\\frac{3}{4}")

    def test_fraction_string_vs_decimal(self):
        assert answers_are_equivalent("1/4", "0.25")

    def test_not_equivalent(self):
        assert not answers_are_equivalent("3", "4")

    def test_variable_eq_vs_plain(self):
        assert answers_are_equivalent("x = 7", "7")

    def test_integer_vs_float(self):
        assert answers_are_equivalent("3", "3.0")
