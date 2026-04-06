"""
Exercise 02: Training Metrics Tracker

During RL/LLM training, we continuously log metrics like loss, reward, KL divergence,
and clip fraction. A good metrics tracker should:

1. Store time-series data for each metric keyed by (name, step).
2. Compute moving averages to smooth out noise.
3. Provide a summary (latest value, mean, min, max for each metric).
4. Detect anomalies:
   - Loss spike: current value > 3x the recent moving average
   - Reward collapse: standard deviation of recent rewards drops to near zero

Reference: slime/utils/wandb_utils.py, tensorboard_utils.py patterns
"""

import numpy as np


class TrainingMetricsTracker:
    """
    A class that tracks training metrics over steps and provides analytics.
    """

    def __init__(self):
        """
        Initialize the tracker.

        TODO: Set up internal data structures to store metrics.
        Each metric is identified by a string name (e.g., "loss", "reward_mean").
        Values are stored as a list of (step, value) tuples.
        """
        raise NotImplementedError("Implement __init__")

    def add_scalar(self, name: str, value: float, step: int) -> None:
        """
        Record a scalar metric value at a given training step.

        Args:
            name: Metric name (e.g., "loss", "reward_mean", "kl_divergence").
            value: The metric value.
            step: The training step number.

        TODO: Implement this method.
        """
        raise NotImplementedError("Implement add_scalar")

    def get_values(self, name: str) -> list[tuple[int, float]]:
        """
        Get all recorded (step, value) pairs for a metric.

        Args:
            name: Metric name.

        Returns:
            List of (step, value) tuples in insertion order.
            Returns empty list if metric not found.

        TODO: Implement this method.
        """
        raise NotImplementedError("Implement get_values")

    def get_moving_average(self, name: str, window: int) -> float | None:
        """
        Compute the moving average of the last `window` values for a metric.

        Args:
            name: Metric name.
            window: Number of recent values to average.

        Returns:
            The moving average as a float, or None if the metric has no values.
            If fewer than `window` values exist, average all available values.

        TODO: Implement this method.
        """
        raise NotImplementedError("Implement get_moving_average")

    def get_summary(self) -> dict[str, dict[str, float]]:
        """
        Get a summary of all tracked metrics.

        Returns:
            Dictionary mapping metric name -> {
                "latest": most recent value,
                "mean": mean of all values,
                "min": minimum value,
                "max": maximum value,
                "count": number of recorded values,
            }

        TODO: Implement this method.
        """
        raise NotImplementedError("Implement get_summary")

    def detect_anomalies(self, window: int = 10) -> list[dict]:
        """
        Detect anomalies in the tracked metrics.

        Anomaly types:
        1. "loss_spike": For any metric containing "loss" in its name,
           if the latest value > 3 * moving_average(window), flag it.
           Note: the moving average includes the current (latest) value in
           its window (i.e., the last `window` values including the newest).
           Only flag if the moving average is > 0.
        2. "reward_collapse": For any metric containing "reward" in its name,
           if the std of the last `window` values < 1e-6, flag it.

        Args:
            window: Window size for moving average / std computation.

        Returns:
            List of anomaly dicts, each with keys:
                - "type": "loss_spike" or "reward_collapse"
                - "metric": the metric name
                - "value": the current/latest value
                - "threshold": the threshold that was exceeded (for loss_spike)
                              or the std value (for reward_collapse)

        TODO: Implement this method.
        """
        raise NotImplementedError("Implement detect_anomalies")
