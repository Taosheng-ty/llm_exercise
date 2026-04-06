"""Tests for Exercise 03: Majority Voting."""

import importlib.util
import os

import torch
import pytest

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("solution_ex03", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
majority_vote = _mod.majority_vote
weighted_majority_vote = _mod.weighted_majority_vote
compute_agreement_rate = _mod.compute_agreement_rate


class TestMajorityVote:
    def test_clear_majority(self):
        assert majority_vote(["A", "B", "A", "C", "A"]) == "A"

    def test_tie_broken_by_first_occurrence(self):
        assert majority_vote(["A", "B", "A", "B"]) == "A"

    def test_single_answer(self):
        assert majority_vote(["X"]) == "X"

    def test_all_empty(self):
        assert majority_vote(["", "", ""]) is None

    def test_empty_list(self):
        assert majority_vote([]) is None

    def test_ignores_empty_strings(self):
        assert majority_vote(["", "A", "", "B", "A"]) == "A"

    def test_all_different(self):
        result = majority_vote(["A", "B", "C"])
        assert result == "A"  # tie, first occurrence wins

    def test_unanimous(self):
        assert majority_vote(["Z", "Z", "Z", "Z"]) == "Z"


class TestWeightedMajorityVote:
    def test_weight_overrides_count(self):
        # B appears once with high weight, A appears twice with low weight
        result = weighted_majority_vote(
            ["A", "B", "A"],
            torch.tensor([0.2, 0.9, 0.2]),
        )
        assert result == "B"  # B: 0.9 > A: 0.4

    def test_weight_reinforces_count(self):
        result = weighted_majority_vote(
            ["A", "B", "A"],
            torch.tensor([0.5, 0.8, 0.5]),
        )
        assert result == "A"  # A: 1.0 > B: 0.8

    def test_empty_answers_ignored(self):
        result = weighted_majority_vote(
            ["", "A", ""],
            torch.tensor([1.0, 0.5, 1.0]),
        )
        assert result == "A"

    def test_all_empty(self):
        result = weighted_majority_vote(
            ["", ""],
            torch.tensor([1.0, 1.0]),
        )
        assert result is None

    def test_tie_broken_by_first_occurrence(self):
        result = weighted_majority_vote(
            ["A", "B"],
            torch.tensor([0.5, 0.5]),
        )
        assert result == "A"


class TestComputeAgreementRate:
    def test_full_agreement(self):
        all_answers = [["A", "A", "A"], ["B", "B", "B"]]
        rate = compute_agreement_rate(all_answers, threshold=0.5)
        assert rate == 1.0

    def test_no_agreement(self):
        all_answers = [["A", "B", "C"], ["X", "Y", "Z"]]
        rate = compute_agreement_rate(all_answers, threshold=0.5)
        assert rate == 0.0

    def test_partial_agreement(self):
        all_answers = [["A", "A", "B"], ["X", "Y", "Z"]]
        rate = compute_agreement_rate(all_answers, threshold=0.5)
        assert abs(rate - 0.5) < 1e-6

    def test_threshold_boundary(self):
        # 2 out of 4 = 0.5, threshold is 0.5 (exclusive >), so no majority
        all_answers = [["A", "A", "B", "B"]]
        rate = compute_agreement_rate(all_answers, threshold=0.5)
        assert rate == 0.0

    def test_empty_input(self):
        assert compute_agreement_rate([]) == 0.0

    def test_questions_with_empty_answers(self):
        all_answers = [["", "", ""]]
        rate = compute_agreement_rate(all_answers, threshold=0.5)
        assert rate == 0.0

    def test_custom_threshold(self):
        all_answers = [["A", "A", "A", "B"]]  # A is 3/4 = 0.75
        assert compute_agreement_rate(all_answers, threshold=0.7) == 1.0
        assert compute_agreement_rate(all_answers, threshold=0.8) == 0.0
