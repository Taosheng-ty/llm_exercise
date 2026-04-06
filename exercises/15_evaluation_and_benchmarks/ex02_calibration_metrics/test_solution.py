"""Tests for Exercise 02: Calibration Metrics (ECE)."""

import importlib.util
import os

import torch
import pytest

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("solution_ex02", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
compute_ece = _mod.compute_ece
classify_calibration = _mod.classify_calibration


class TestComputeECE:
    def test_perfectly_calibrated(self):
        """A model that is always 100% confident and always correct has ECE=0."""
        probs = torch.tensor([
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ])
        labels = torch.tensor([1, 1, 0, 0])
        ece, bin_stats = compute_ece(probs, labels, n_bins=10)
        assert ece.item() < 1e-6

    def test_overconfident_model(self):
        """Model always says 0.9 but is only correct half the time -> large ECE."""
        probs = torch.tensor([
            [0.1, 0.9],
            [0.1, 0.9],
            [0.1, 0.9],
            [0.1, 0.9],
        ])
        labels = torch.tensor([1, 1, 0, 0])
        ece, bin_stats = compute_ece(probs, labels, n_bins=10)
        # confidence=0.9, accuracy=0.5 -> gap=0.4, all samples in one bin
        assert abs(ece.item() - 0.4) < 1e-5

    def test_bin_stats_structure(self):
        """Check that bin_stats has correct structure."""
        probs = torch.tensor([[0.3, 0.7], [0.6, 0.4]])
        labels = torch.tensor([1, 0])
        ece, bin_stats = compute_ece(probs, labels, n_bins=5)
        assert len(bin_stats) == 5
        for stat in bin_stats:
            assert "bin_lower" in stat
            assert "bin_upper" in stat
            assert "count" in stat
            assert "avg_confidence" in stat
            assert "avg_accuracy" in stat
            assert "gap" in stat

    def test_all_wrong_high_confidence(self):
        """All wrong with high confidence -> ECE near confidence level."""
        probs = torch.tensor([
            [0.0, 1.0],
            [0.0, 1.0],
        ])
        labels = torch.tensor([0, 0])  # all wrong
        ece, _ = compute_ece(probs, labels, n_bins=10)
        assert abs(ece.item() - 1.0) < 1e-5

    def test_empty_bins(self):
        """Bins with no samples should have zero gap."""
        probs = torch.tensor([[0.1, 0.9]])
        labels = torch.tensor([1])
        ece, bin_stats = compute_ece(probs, labels, n_bins=10)
        empty_bins = [s for s in bin_stats if s["count"] == 0]
        assert len(empty_bins) > 0
        for s in empty_bins:
            assert s["gap"] == 0.0

    def test_total_count_matches(self):
        """Sum of bin counts should equal total samples."""
        probs = torch.rand(50, 3)
        probs = probs / probs.sum(dim=1, keepdim=True)
        labels = torch.randint(0, 3, (50,))
        _, bin_stats = compute_ece(probs, labels, n_bins=10)
        total = sum(s["count"] for s in bin_stats)
        assert total == 50


class TestClassifyCalibration:
    def test_well_calibrated(self):
        assert classify_calibration(0.02) == "well-calibrated"
        assert classify_calibration(0.0) == "well-calibrated"

    def test_moderately_calibrated(self):
        assert classify_calibration(0.05) == "moderately-calibrated"
        assert classify_calibration(0.10) == "moderately-calibrated"

    def test_poorly_calibrated(self):
        assert classify_calibration(0.15) == "poorly-calibrated"
        assert classify_calibration(0.50) == "poorly-calibrated"
