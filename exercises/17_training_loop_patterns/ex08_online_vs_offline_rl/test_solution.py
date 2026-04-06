"""Tests for Exercise 08: Online vs Offline RL Data Flow"""

import importlib.util
import os
import numpy as np
import pytest

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
OnlineScheduler = _mod.OnlineScheduler
OfflineScheduler = _mod.OfflineScheduler
AsyncScheduler = _mod.AsyncScheduler


class TestOnlineScheduler:
    def test_zero_staleness(self):
        """Online training should always have 0 staleness."""
        sched = OnlineScheduler()
        result = sched.run(10)
        assert all(s == 0 for s in result["staleness_history"])
        assert result["avg_staleness"] == 0.0

    def test_data_log_length(self):
        sched = OnlineScheduler()
        result = sched.run(5)
        assert len(result["data_log"]) == 5

    def test_data_versions_match_policy(self):
        """Each step's data should come from the current policy version."""
        sched = OnlineScheduler()
        result = sched.run(5)
        for step, data_ver in result["data_log"]:
            assert data_ver == step  # data version = policy version at that step

    def test_policy_version_increments(self):
        sched = OnlineScheduler()
        sched.run(10)
        assert sched.policy_version == 10


class TestOfflineScheduler:
    def test_increasing_staleness(self):
        """Offline training staleness should increase linearly."""
        sched = OfflineScheduler()
        result = sched.run(10)
        expected_staleness = list(range(10))
        assert result["staleness_history"] == expected_staleness

    def test_data_always_from_version_zero(self):
        sched = OfflineScheduler()
        result = sched.run(5)
        for step, data_ver in result["data_log"]:
            assert data_ver == 0

    def test_avg_staleness(self):
        sched = OfflineScheduler()
        result = sched.run(10)
        # mean of 0..9 = 4.5
        assert result["avg_staleness"] == pytest.approx(4.5)

    def test_policy_version_increments(self):
        sched = OfflineScheduler()
        sched.run(10)
        assert sched.policy_version == 10


class TestAsyncScheduler:
    def test_update_interval_1_staleness(self):
        """With update_interval=1, data is 1 step behind after first step."""
        sched = AsyncScheduler(update_interval=1)
        result = sched.run(5)
        # Step 0: data from v0, policy=0, staleness=0
        # Step 1: data from v1 (refreshed after step 0), policy=1, staleness=0
        # ...
        # With interval=1 and refresh after each step, staleness should be 0
        assert all(s == 0 for s in result["staleness_history"])

    def test_update_interval_2(self):
        """With update_interval=2, data refreshes every 2 steps."""
        sched = AsyncScheduler(update_interval=2)
        result = sched.run(6)
        # Step 0: data_gen=0, policy=0, staleness=0, then policy->1
        # Step 1: data_gen=0, policy=1, staleness=1, then policy->2, refresh: data_gen=2
        # Step 2: data_gen=2, policy=2, staleness=0, then policy->3
        # Step 3: data_gen=2, policy=3, staleness=1, then policy->4, refresh: data_gen=4
        # Step 4: data_gen=4, policy=4, staleness=0, then policy->5
        # Step 5: data_gen=4, policy=5, staleness=1, then policy->6, refresh: data_gen=6
        assert result["staleness_history"] == [0, 1, 0, 1, 0, 1]

    def test_async_less_stale_than_offline(self):
        """Async training should have less staleness than offline."""
        online = OnlineScheduler().run(20)
        offline = OfflineScheduler().run(20)
        async_1 = AsyncScheduler(update_interval=1).run(20)
        async_3 = AsyncScheduler(update_interval=3).run(20)

        assert online["avg_staleness"] <= async_1["avg_staleness"]
        assert async_1["avg_staleness"] <= async_3["avg_staleness"]
        assert async_3["avg_staleness"] < offline["avg_staleness"]

    def test_data_log_length(self):
        sched = AsyncScheduler(update_interval=3)
        result = sched.run(10)
        assert len(result["data_log"]) == 10

    def test_policy_version_increments(self):
        sched = AsyncScheduler(update_interval=2)
        sched.run(10)
        assert sched.policy_version == 10

    def test_large_interval(self):
        """With update_interval > num_steps, should behave like offline."""
        sched = AsyncScheduler(update_interval=100)
        result = sched.run(10)
        # Data always from version 0 since no refresh happens
        for step, data_ver in result["data_log"]:
            assert data_ver == 0
        assert result["staleness_history"] == list(range(10))


class TestComparison:
    def test_online_best_offline_worst(self):
        """Online should have lowest staleness, offline highest."""
        n = 50
        online = OnlineScheduler().run(n)
        offline = OfflineScheduler().run(n)
        async_s = AsyncScheduler(update_interval=5).run(n)

        assert online["avg_staleness"] == 0.0
        assert async_s["avg_staleness"] < offline["avg_staleness"]
        assert offline["avg_staleness"] > 0
