"""Tests for Exercise 06: Curriculum Learning Scheduler"""

import importlib.util
import os
import numpy as np
import pytest

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
CurriculumScheduler = _mod.CurriculumScheduler


class TestCurriculumScheduler:
    def test_linear_step_zero(self):
        """At step 0, only difficulty=0 samples should be available."""
        diffs = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
        sched = CurriculumScheduler(diffs, total_steps=100)
        indices = sched.get_available_indices(0, "linear")
        np.testing.assert_array_equal(indices, [0])

    def test_linear_step_end(self):
        """At total_steps, all samples should be available."""
        diffs = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
        sched = CurriculumScheduler(diffs, total_steps=100)
        indices = sched.get_available_indices(100, "linear")
        np.testing.assert_array_equal(indices, [0, 1, 2, 3, 4])

    def test_linear_midpoint(self):
        """At step 50/100, competence=0.5, samples with diff <= 0.5 available."""
        diffs = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
        sched = CurriculumScheduler(diffs, total_steps=100)
        indices = sched.get_available_indices(50, "linear")
        np.testing.assert_array_equal(indices, [0, 1, 2])

    def test_exponential_starts_slow(self):
        """Exponential curriculum should start slowly."""
        diffs = np.linspace(0, 1, 11)
        sched = CurriculumScheduler(diffs, total_steps=100)
        c_exp = sched.get_competence(10, "exponential")
        c_lin = sched.get_competence(10, "linear")
        # Exponential should have higher competence early (1 - exp(-0.5) ~ 0.39 vs 0.1)
        assert c_exp > c_lin

    def test_exponential_reaches_one(self):
        """Exponential should reach 1.0 at total_steps."""
        sched = CurriculumScheduler(np.array([0.0]), total_steps=100)
        c = sched.get_competence(100, "exponential")
        assert c == pytest.approx(1.0, abs=0.01)

    def test_competence_sqrt(self):
        """Competence-based should use sqrt."""
        sched = CurriculumScheduler(np.array([0.0]), total_steps=100)
        c = sched.get_competence(25, "competence")
        assert c == pytest.approx(0.5, rel=1e-5)

    def test_competence_at_zero(self):
        sched = CurriculumScheduler(np.array([0.0, 0.5, 1.0]), total_steps=100)
        c = sched.get_competence(0, "competence")
        assert c == pytest.approx(0.0)

    def test_monotonic_increase(self):
        """Available indices should be non-decreasing over steps."""
        diffs = np.random.default_rng(42).random(50)
        sched = CurriculumScheduler(diffs, total_steps=100)
        prev_count = 0
        for strategy in ["linear", "exponential", "competence"]:
            prev_count = 0
            for step in range(0, 101, 10):
                indices = sched.get_available_indices(step, strategy)
                assert len(indices) >= prev_count, f"{strategy} at step {step}"
                prev_count = len(indices)

    def test_all_strategies_include_all_at_end(self):
        """All strategies should include all data at total_steps."""
        diffs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        sched = CurriculumScheduler(diffs, total_steps=100)
        for strategy in ["linear", "exponential", "competence"]:
            indices = sched.get_available_indices(100, strategy)
            assert len(indices) == 5, f"{strategy} should include all at end"

    def test_unknown_strategy_raises(self):
        sched = CurriculumScheduler(np.array([0.5]), total_steps=100)
        with pytest.raises(ValueError):
            sched.get_competence(50, "unknown")
