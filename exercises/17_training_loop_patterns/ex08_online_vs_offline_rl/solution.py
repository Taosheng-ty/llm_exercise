"""
Solution for Exercise 08: Online vs Offline RL Data Flow
"""

import numpy as np


class OnlineScheduler:
    """Synchronous online RL: generate with current policy, then train."""

    def __init__(self):
        self.policy_version = 0

    def run(self, num_steps: int) -> dict:
        data_log = []
        staleness_history = []

        for step in range(num_steps):
            # Generate with current policy
            data_policy_version = self.policy_version
            data_log.append((step, data_policy_version))
            staleness = self.policy_version - data_policy_version
            staleness_history.append(staleness)
            # Train and update
            self.policy_version += 1

        avg_staleness = float(np.mean(staleness_history)) if staleness_history else 0.0
        return {
            "data_log": data_log,
            "staleness_history": staleness_history,
            "avg_staleness": avg_staleness,
        }


class OfflineScheduler:
    """Offline RL: train multiple epochs on fixed data from initial policy."""

    def __init__(self):
        self.policy_version = 0

    def run(self, num_steps: int) -> dict:
        data_log = []
        staleness_history = []

        for step in range(num_steps):
            # Data always from version 0
            data_policy_version = 0
            data_log.append((step, data_policy_version))
            staleness = self.policy_version - data_policy_version
            staleness_history.append(staleness)
            # Train and update
            self.policy_version += 1

        avg_staleness = float(np.mean(staleness_history)) if staleness_history else 0.0
        return {
            "data_log": data_log,
            "staleness_history": staleness_history,
            "avg_staleness": avg_staleness,
        }


class AsyncScheduler:
    """Async RL: overlap generation with training (like train_async.py)."""

    def __init__(self, update_interval: int = 1):
        self.policy_version = 0
        self.update_interval = update_interval

    def run(self, num_steps: int) -> dict:
        data_log = []
        staleness_history = []

        # Initial generation happens at policy_version 0
        data_gen_version = 0

        for step in range(num_steps):
            # Train on data from data_gen_version
            data_log.append((step, data_gen_version))
            staleness = self.policy_version - data_gen_version
            staleness_history.append(staleness)

            # Train and update policy
            self.policy_version += 1

            # After every update_interval steps, refresh data
            if (step + 1) % self.update_interval == 0:
                data_gen_version = self.policy_version

        avg_staleness = float(np.mean(staleness_history)) if staleness_history else 0.0
        return {
            "data_log": data_log,
            "staleness_history": staleness_history,
            "avg_staleness": avg_staleness,
        }
