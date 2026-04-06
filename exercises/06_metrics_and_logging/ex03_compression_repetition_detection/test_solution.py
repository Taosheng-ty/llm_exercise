"""Tests for Exercise 03: Compression-Based Repetition Detection."""

import importlib
import os

import pytest

# Import solution from the same directory as this test file
_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("solution_ex03", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
compression_ratio = _mod.compression_ratio
detect_repetition = _mod.detect_repetition
has_repetition = _mod.has_repetition
ngram_repetition_fraction = _mod.ngram_repetition_fraction


class TestCompressionRatio:
    def test_empty_string(self):
        assert compression_ratio("") == 0.0

    def test_repetitive_text_high_ratio(self):
        text = "abc " * 5000
        ratio = compression_ratio(text)
        assert ratio > 10.0  # Highly repetitive => high ratio

    def test_random_text_low_ratio(self):
        import string
        import random
        random.seed(42)
        text = "".join(random.choices(string.ascii_letters + string.digits, k=5000))
        ratio = compression_ratio(text)
        assert ratio < 3.0  # Random text doesn't compress well

    def test_single_char(self):
        ratio = compression_ratio("a")
        # Single byte compressed with zlib overhead => ratio < 1
        assert ratio > 0.0

    def test_moderate_repetition(self):
        # Some repetition but not extreme
        text = ("hello world. this is a test sentence. " * 50)
        ratio = compression_ratio(text)
        assert ratio > 5.0


class TestHasRepetition:
    def test_short_text_never_repetitive(self):
        """Texts shorter than tail_length should return False."""
        text = "abc " * 100
        assert has_repetition(text, tail_length=10000) is False

    def test_long_repetitive_text(self):
        text = "repeat " * 20000  # ~140k chars, very repetitive
        assert has_repetition(text, threshold=10.0, tail_length=10000) is True

    def test_long_varied_text(self):
        import string
        import random
        random.seed(123)
        text = "".join(random.choices(string.ascii_letters + " ", k=20000))
        assert has_repetition(text, threshold=10.0, tail_length=10000) is False

    def test_custom_threshold(self):
        text = "abc " * 20000
        # With very high threshold, even repetitive text might not trigger
        assert has_repetition(text, threshold=1000.0, tail_length=10000) is False


class TestNgramRepetitionFraction:
    def test_text_shorter_than_n(self):
        assert ngram_repetition_fraction("ab", n=3) == 0.0

    def test_all_unique_ngrams(self):
        # "abcde" with n=3: "abc", "bcd", "cde" - all unique => 0.0
        assert ngram_repetition_fraction("abcde", n=3) == pytest.approx(0.0)

    def test_fully_repetitive(self):
        # "aaaaaa" with n=3: "aaa","aaa","aaa","aaa" - 4 total, 1 unique => 3/4
        result = ngram_repetition_fraction("aaaaaa", n=3)
        assert result == pytest.approx(0.75)

    def test_moderate_repetition(self):
        text = "abcabc"
        # n=3: "abc","bca","cab","abc" => 4 total, 3 unique => 1/4 = 0.25
        assert ngram_repetition_fraction(text, n=3) == pytest.approx(0.25)

    def test_empty_string(self):
        assert ngram_repetition_fraction("", n=3) == 0.0


class TestDetectRepetition:
    def test_highly_repetitive(self):
        text = "xyz " * 5000
        result = detect_repetition(text, compression_threshold=10.0, ngram_threshold=0.7)
        assert result["is_repetitive"] is True
        assert result["compression_ratio"] > 10.0
        assert result["ngram_repetition"] > 0.7

    def test_non_repetitive(self):
        import string
        import random
        random.seed(99)
        text = "".join(random.choices(string.ascii_letters + string.digits + " ", k=5000))
        result = detect_repetition(text, compression_threshold=10.0, ngram_threshold=0.7)
        assert result["is_repetitive"] is False

    def test_result_keys(self):
        result = detect_repetition("hello world")
        expected_keys = {
            "compression_ratio", "ngram_repetition",
            "is_repetitive_compression", "is_repetitive_ngram", "is_repetitive",
        }
        assert set(result.keys()) == expected_keys

    def test_compression_only_trigger(self):
        """Test where only compression detects repetition."""
        text = "ab " * 5000
        result = detect_repetition(
            text, compression_threshold=5.0, ngram_threshold=0.999
        )
        assert result["is_repetitive_compression"] is True
        assert result["is_repetitive"] is True
