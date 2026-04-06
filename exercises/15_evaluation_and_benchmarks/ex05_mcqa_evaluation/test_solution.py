"""Tests for Exercise 05: MCQA Evaluation."""

import importlib.util
import os

import pytest

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("solution_ex05", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
extract_answer_letter = _mod.extract_answer_letter
text_similarity_fallback = _mod.text_similarity_fallback
evaluate_mcqa_dataset = _mod.evaluate_mcqa_dataset


class TestExtractAnswerLetter:
    def test_answer_is_pattern(self):
        assert extract_answer_letter("The answer is B.", ["A", "B", "C", "D"]) == "B"

    def test_option_is_pattern(self):
        assert extract_answer_letter("The option is C", ["A", "B", "C", "D"]) == "C"

    def test_choice_colon_pattern(self):
        assert extract_answer_letter("My choice: A", ["A", "B", "C"]) == "A"

    def test_correct_pattern(self):
        assert extract_answer_letter("D is the correct answer", ["A", "B", "C", "D"]) == "D"

    def test_final_answer_pattern(self):
        assert extract_answer_letter("My final answer is B", ["A", "B", "C", "D"]) == "B"

    def test_fallback_last_letter(self):
        assert extract_answer_letter("Let me think... C", ["A", "B", "C", "D"]) == "C"

    def test_chain_of_thought_stripped(self):
        response = "Let me think about A... </think> The answer is B."
        assert extract_answer_letter(response, ["A", "B", "C", "D"]) == "B"

    def test_invalid_letter_skipped(self):
        assert extract_answer_letter("The answer is Z", ["A", "B", "C", "D"]) is None

    def test_none_response(self):
        assert extract_answer_letter("", ["A", "B"]) is None

    def test_default_valid_letters(self):
        assert extract_answer_letter("The answer is E") == "E"

    def test_no_match(self):
        # Only letters A-D are valid, and none appear as standalone uppercase
        assert extract_answer_letter("i have no idea about this question.", ["A", "B", "C", "D"]) is None


class TestTextSimilarityFallback:
    def test_exact_match(self):
        idx = text_similarity_fallback(
            "photosynthesis",
            ["mitosis", "photosynthesis", "osmosis"],
        )
        assert idx == 1

    def test_partial_overlap(self):
        idx = text_similarity_fallback(
            "the process of photosynthesis in plants",
            ["cell division mitosis", "photosynthesis plants", "water osmosis"],
        )
        assert idx == 1

    def test_empty_response(self):
        idx = text_similarity_fallback("", ["a", "b", "c"])
        assert idx == 0

    def test_no_overlap(self):
        idx = text_similarity_fallback("xyz", ["aaa", "bbb", "ccc"])
        assert idx == 0

    def test_tie_broken_by_index(self):
        idx = text_similarity_fallback("cat", ["cat", "cat", "dog"])
        assert idx == 0


class TestEvaluateMcqaDataset:
    def test_all_correct_extraction(self):
        responses = ["The answer is A", "The answer is B", "The answer is C"]
        correct = ["A", "B", "C"]
        result = evaluate_mcqa_dataset(responses, correct)
        assert result["accuracy"] == 1.0
        assert result["extracted_count"] == 3
        assert result["fallback_count"] == 0
        assert result["failed_count"] == 0
        assert result["total"] == 3

    def test_all_wrong(self):
        responses = ["The answer is A", "The answer is A", "The answer is A"]
        correct = ["B", "C", "D"]
        result = evaluate_mcqa_dataset(responses, correct)
        assert result["accuracy"] == 0.0

    def test_fallback_used(self):
        responses = ["I think photosynthesis is the answer"]
        correct = ["B"]
        choices = [["mitosis", "photosynthesis", "osmosis", "diffusion"]]
        result = evaluate_mcqa_dataset(responses, correct, choices_list=choices)
        assert result["fallback_count"] == 1
        assert result["accuracy"] == 1.0

    def test_failed_without_choices(self):
        responses = ["I have absolutely no idea"]
        correct = ["A"]
        result = evaluate_mcqa_dataset(responses, correct)
        assert result["failed_count"] == 1
        assert result["accuracy"] == 0.0

    def test_mixed(self):
        responses = [
            "The answer is A",
            "I think photosynthesis",
            "No clue whatsoever",
        ]
        correct = ["A", "B", "C"]
        choices = [
            ["alpha", "beta", "gamma", "delta"],
            ["mitosis", "photosynthesis", "osmosis", "diffusion"],
            ["x", "y", "z", "w"],
        ]
        result = evaluate_mcqa_dataset(responses, correct, choices_list=choices)
        assert result["total"] == 3
        assert result["extracted_count"] == 1
        # Second uses fallback, third uses fallback
        assert result["fallback_count"] == 2
        assert result["failed_count"] == 0

    def test_empty_dataset(self):
        result = evaluate_mcqa_dataset([], [])
        assert result["accuracy"] == 0.0
        assert result["total"] == 0
