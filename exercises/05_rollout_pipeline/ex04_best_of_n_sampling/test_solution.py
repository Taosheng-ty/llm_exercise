"""Tests for Exercise 04: Best-of-N Sampling."""

import importlib.util
import os

import pytest

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

Response = _mod.Response
SelectionResult = _mod.SelectionResult
BatchSelectionStats = _mod.BatchSelectionStats
greedy_best = _mod.greedy_best
weighted_sample = _mod.weighted_sample
rejection_sample = _mod.rejection_sample
batch_best_of_n = _mod.batch_best_of_n


def _responses(scores):
    return [Response(text=f"resp_{i}", score=s) for i, s in enumerate(scores)]


def test_greedy_picks_best():
    resps = _responses([0.1, 0.9, 0.5])
    result = greedy_best(resps)
    assert result.selected.score == 0.9
    assert result.method == "greedy"
    assert result.accepted


def test_greedy_empty_raises():
    with pytest.raises(ValueError):
        greedy_best([])


def test_weighted_sample_valid():
    resps = _responses([1.0, 2.0, 3.0])
    result = weighted_sample(resps, seed=0)
    assert result.selected in resps
    assert result.method == "weighted"
    assert result.accepted


def test_weighted_sample_bias():
    """The highest-scored response should be selected most often."""
    resps = _responses([0.0, 0.0, 100.0])
    counts = [0, 0, 0]
    for seed in range(500):
        result = weighted_sample(resps, seed=seed)
        idx = resps.index(result.selected)
        counts[idx] += 1
    # Response with score 100 should dominate
    assert counts[2] > 400


def test_rejection_accept():
    resps = _responses([0.3, 0.8, 0.6])
    result = rejection_sample(resps, threshold=0.7)
    assert result.accepted
    assert result.selected.score == 0.8


def test_rejection_reject():
    resps = _responses([0.1, 0.2, 0.3])
    result = rejection_sample(resps, threshold=0.5)
    assert not result.accepted
    assert result.selected is None
    assert result.method == "rejection"


def test_batch_greedy():
    groups = [_responses([0.1, 0.9]), _responses([0.5, 0.3])]
    results, stats = batch_best_of_n(groups, method="greedy")
    assert len(results) == 2
    assert stats.total_prompts == 2
    assert stats.accepted_count == 2
    assert stats.acceptance_rate == 1.0
    assert results[0].selected.score == 0.9
    assert results[1].selected.score == 0.5


def test_batch_rejection_stats():
    groups = [
        _responses([0.1, 0.2]),  # max=0.2, below 0.5 -> reject
        _responses([0.6, 0.8]),  # max=0.8, above 0.5 -> accept
        _responses([0.4, 0.3]),  # max=0.4, below 0.5 -> reject
    ]
    results, stats = batch_best_of_n(groups, method="rejection", threshold=0.5)
    assert stats.accepted_count == 1
    assert stats.rejected_count == 2
    assert stats.acceptance_rate == pytest.approx(1 / 3)
    assert stats.mean_selected_score == pytest.approx(0.8)


def test_batch_unknown_method_raises():
    with pytest.raises(ValueError):
        batch_best_of_n([], method="invalid")
