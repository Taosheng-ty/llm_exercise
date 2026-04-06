"""Tests for Exercise 07: Training State Manager"""

import importlib.util
import os
import numpy as np
import pytest

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TrainingStateManager = _mod.TrainingStateManager


class TestTrainingStateManager:
    def test_initial_state(self):
        mgr = TrainingStateManager(patience=5)
        assert mgr.current_step == 0
        assert mgr.current_epoch == 0
        assert mgr.best_metric is None
        assert len(mgr.loss_history) == 0

    def test_step_increments(self):
        mgr = TrainingStateManager()
        mgr.step(loss=0.5, lr=1e-3)
        mgr.step(loss=0.4, lr=9e-4)
        assert mgr.current_step == 2
        assert len(mgr.loss_history) == 2
        assert len(mgr.lr_history) == 2
        assert mgr.loss_history == [0.5, 0.4]

    def test_next_epoch(self):
        mgr = TrainingStateManager()
        mgr.next_epoch()
        mgr.next_epoch()
        assert mgr.current_epoch == 2

    def test_metric_tracking_higher_is_better(self):
        mgr = TrainingStateManager(patience=3, metric_higher_is_better=True)
        mgr.step(0.5, 1e-3)
        mgr.update_metric(0.7)
        assert mgr.best_metric == 0.7
        mgr.step(0.4, 1e-3)
        mgr.update_metric(0.8)
        assert mgr.best_metric == 0.8
        assert mgr.steps_without_improvement == 0

    def test_metric_tracking_lower_is_better(self):
        mgr = TrainingStateManager(patience=3, metric_higher_is_better=False)
        mgr.update_metric(1.0)
        assert mgr.best_metric == 1.0
        mgr.update_metric(0.5)
        assert mgr.best_metric == 0.5
        mgr.update_metric(0.8)
        assert mgr.steps_without_improvement == 1

    def test_early_stopping_triggers(self):
        mgr = TrainingStateManager(patience=3)
        mgr.update_metric(0.9)  # best
        mgr.update_metric(0.8)  # no improvement
        mgr.update_metric(0.7)  # no improvement
        assert not mgr.should_stop()
        mgr.update_metric(0.6)  # 3 steps without improvement
        assert mgr.should_stop()

    def test_early_stopping_resets(self):
        mgr = TrainingStateManager(patience=3)
        mgr.update_metric(0.5)
        mgr.update_metric(0.4)  # no improve
        mgr.update_metric(0.3)  # no improve
        assert not mgr.should_stop()
        mgr.update_metric(0.9)  # new best - resets counter
        assert mgr.steps_without_improvement == 0
        assert not mgr.should_stop()

    def test_state_dict_roundtrip(self):
        mgr = TrainingStateManager(patience=5, metric_higher_is_better=False)
        for i in range(10):
            mgr.step(loss=1.0 / (i + 1), lr=1e-3 * (0.9 ** i))
        mgr.next_epoch()
        mgr.update_metric(0.5)
        mgr.update_metric(0.3)

        sd = mgr.state_dict()

        mgr2 = TrainingStateManager()
        mgr2.load_state_dict(sd)

        assert mgr2.current_step == 10
        assert mgr2.current_epoch == 1
        assert mgr2.best_metric == 0.3
        assert mgr2.patience == 5
        assert mgr2.metric_higher_is_better is False
        assert len(mgr2.loss_history) == 10
        assert len(mgr2.metric_history) == 2

    def test_get_summary(self):
        mgr = TrainingStateManager()
        for i in range(20):
            mgr.step(loss=2.0 - i * 0.1, lr=1e-3)
        mgr.next_epoch()
        mgr.update_metric(0.85)

        summary = mgr.get_summary()
        assert summary["total_steps"] == 20
        assert summary["total_epochs"] == 1
        assert summary["best_metric"] == 0.85
        assert summary["final_loss"] == pytest.approx(0.1, abs=0.01)
        # avg of last 10 losses: 2.0-1.0=1.0, 2.0-1.1=0.9, ..., 2.0-1.9=0.1
        assert summary["avg_loss_last_10"] is not None

    def test_get_summary_empty(self):
        mgr = TrainingStateManager()
        summary = mgr.get_summary()
        assert summary["total_steps"] == 0
        assert summary["final_loss"] is None
        assert summary["avg_loss_last_10"] is None

    def test_no_early_stop_without_metrics(self):
        """Should not stop if no metrics have been recorded."""
        mgr = TrainingStateManager(patience=3)
        assert not mgr.should_stop()
