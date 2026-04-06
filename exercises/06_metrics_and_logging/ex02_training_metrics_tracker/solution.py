"""
Exercise 02: Training Metrics Tracker - Solution

Reference: slime/utils/wandb_utils.py, tensorboard_utils.py patterns
"""

import numpy as np
from collections import defaultdict


class TrainingMetricsTracker:
    """
    A class that tracks training metrics over steps and provides analytics.
    """

    def __init__(self):
        # Maps metric name -> list of (step, value) tuples
        self._data: dict[str, list[tuple[int, float]]] = defaultdict(list)

    def add_scalar(self, name: str, value: float, step: int) -> None:
        self._data[name].append((step, value))

    def get_values(self, name: str) -> list[tuple[int, float]]:
        return list(self._data.get(name, []))

    def get_moving_average(self, name: str, window: int) -> float | None:
        entries = self._data.get(name, [])
        if not entries:
            return None
        values = [v for _, v in entries]
        recent = values[-window:]
        return float(np.mean(recent))

    def get_summary(self) -> dict[str, dict[str, float]]:
        summary = {}
        for name, entries in self._data.items():
            values = [v for _, v in entries]
            arr = np.array(values)
            summary[name] = {
                "latest": values[-1],
                "mean": float(np.mean(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "count": len(values),
            }
        return summary

    def detect_anomalies(self, window: int = 10) -> list[dict]:
        anomalies = []

        for name, entries in self._data.items():
            values = [v for _, v in entries]
            if not values:
                continue

            # Loss spike detection
            if "loss" in name:
                ma = self.get_moving_average(name, window)
                latest = values[-1]
                if ma is not None and ma > 0 and latest > 3 * ma:
                    anomalies.append({
                        "type": "loss_spike",
                        "metric": name,
                        "value": latest,
                        "threshold": 3 * ma,
                    })

            # Reward collapse detection
            if "reward" in name:
                recent = values[-window:]
                std = float(np.std(recent))
                if std < 1e-6:
                    anomalies.append({
                        "type": "reward_collapse",
                        "metric": name,
                        "value": values[-1],
                        "threshold": std,
                    })

        return anomalies
