"""Tests for Exercise 02: Training Metrics Tracker."""

import importlib
import os

import numpy as np
import pytest

# Import solution from the same directory as this test file
_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("solution_ex02", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TrainingMetricsTracker = _mod.TrainingMetricsTracker


class TestAddAndGetValues:
    def test_add_and_retrieve(self):
        tracker = TrainingMetricsTracker()
        tracker.add_scalar("loss", 2.5, step=0)
        tracker.add_scalar("loss", 2.0, step=1)
        values = tracker.get_values("loss")
        assert values == [(0, 2.5), (1, 2.0)]

    def test_unknown_metric_returns_empty(self):
        tracker = TrainingMetricsTracker()
        assert tracker.get_values("nonexistent") == []

    def test_multiple_metrics(self):
        tracker = TrainingMetricsTracker()
        tracker.add_scalar("loss", 1.0, step=0)
        tracker.add_scalar("reward_mean", 0.5, step=0)
        assert len(tracker.get_values("loss")) == 1
        assert len(tracker.get_values("reward_mean")) == 1


class TestMovingAverage:
    def test_basic_moving_average(self):
        tracker = TrainingMetricsTracker()
        for i in range(10):
            tracker.add_scalar("loss", float(i), step=i)
        # Last 5 values: 5,6,7,8,9 => mean = 7.0
        assert tracker.get_moving_average("loss", 5) == pytest.approx(7.0)

    def test_window_larger_than_data(self):
        tracker = TrainingMetricsTracker()
        tracker.add_scalar("loss", 4.0, step=0)
        tracker.add_scalar("loss", 6.0, step=1)
        # Only 2 values, window=10 => average all: (4+6)/2 = 5.0
        assert tracker.get_moving_average("loss", 10) == pytest.approx(5.0)

    def test_no_data_returns_none(self):
        tracker = TrainingMetricsTracker()
        assert tracker.get_moving_average("loss", 5) is None

    def test_window_1(self):
        tracker = TrainingMetricsTracker()
        tracker.add_scalar("loss", 1.0, step=0)
        tracker.add_scalar("loss", 3.0, step=1)
        assert tracker.get_moving_average("loss", 1) == pytest.approx(3.0)


class TestGetSummary:
    def test_summary_structure(self):
        tracker = TrainingMetricsTracker()
        tracker.add_scalar("loss", 2.0, step=0)
        tracker.add_scalar("loss", 4.0, step=1)
        tracker.add_scalar("loss", 3.0, step=2)
        summary = tracker.get_summary()
        assert "loss" in summary
        s = summary["loss"]
        assert s["latest"] == pytest.approx(3.0)
        assert s["mean"] == pytest.approx(3.0)
        assert s["min"] == pytest.approx(2.0)
        assert s["max"] == pytest.approx(4.0)
        assert s["count"] == 3

    def test_empty_tracker(self):
        tracker = TrainingMetricsTracker()
        assert tracker.get_summary() == {}


class TestDetectAnomalies:
    def test_loss_spike_detected(self):
        tracker = TrainingMetricsTracker()
        # Normal losses around 1.0
        for i in range(10):
            tracker.add_scalar("loss", 1.0, step=i)
        # Sudden spike
        tracker.add_scalar("loss", 10.0, step=10)
        anomalies = tracker.detect_anomalies(window=10)
        assert len(anomalies) == 1
        assert anomalies[0]["type"] == "loss_spike"
        assert anomalies[0]["metric"] == "loss"
        assert anomalies[0]["value"] == 10.0

    def test_no_loss_spike_for_normal_values(self):
        tracker = TrainingMetricsTracker()
        for i in range(10):
            tracker.add_scalar("loss", 1.0 + 0.1 * i, step=i)
        anomalies = tracker.detect_anomalies(window=10)
        loss_spikes = [a for a in anomalies if a["type"] == "loss_spike"]
        assert len(loss_spikes) == 0

    def test_reward_collapse_detected(self):
        tracker = TrainingMetricsTracker()
        # All rewards the same => std ~ 0
        for i in range(10):
            tracker.add_scalar("reward_mean", 0.5, step=i)
        anomalies = tracker.detect_anomalies(window=10)
        collapse = [a for a in anomalies if a["type"] == "reward_collapse"]
        assert len(collapse) == 1
        assert collapse[0]["metric"] == "reward_mean"

    def test_no_reward_collapse_with_variance(self):
        tracker = TrainingMetricsTracker()
        for i in range(10):
            tracker.add_scalar("reward_mean", float(i), step=i)
        anomalies = tracker.detect_anomalies(window=10)
        collapse = [a for a in anomalies if a["type"] == "reward_collapse"]
        assert len(collapse) == 0

    def test_unrelated_metric_no_anomaly(self):
        """Metrics without 'loss' or 'reward' should not trigger anomalies."""
        tracker = TrainingMetricsTracker()
        for i in range(10):
            tracker.add_scalar("kl_divergence", 1.0, step=i)
        tracker.add_scalar("kl_divergence", 100.0, step=10)
        anomalies = tracker.detect_anomalies(window=10)
        assert len(anomalies) == 0
