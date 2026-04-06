"""
Exercise 07: Training State Manager

Difficulty: Medium
Framework: numpy

Background:
    During training, we need to track state for checkpointing, logging,
    and early stopping. The state manager tracks:
    - current_step, current_epoch
    - best_metric and which step achieved it
    - lr_history and loss_history
    - Early stopping: stop if the metric doesn't improve for patience steps

    Reference: slime's training loop (train.py) manages rollout_id,
    save intervals, and evaluation checkpointing.

Implement the TrainingStateManager class:
    - step(loss, lr): Record one training step
    - update_metric(metric): Record evaluation metric (higher is better)
    - should_stop() -> bool: Check early stopping condition
    - state_dict() -> dict: Serialize state for checkpointing
    - load_state_dict(d): Restore from checkpoint
    - get_summary() -> dict: Return training summary statistics
"""

import numpy as np


class TrainingStateManager:
    """Manages training state for checkpointing and early stopping."""

    def __init__(self, patience: int = 10, metric_higher_is_better: bool = True):
        """
        Args:
            patience: number of metric evaluations without improvement before stopping
            metric_higher_is_better: if True, higher metric = better
        """
        # TODO: Initialize state variables
        # - current_step = 0
        # - current_epoch = 0
        # - best_metric = None
        # - best_metric_step = 0
        # - steps_without_improvement = 0
        # - loss_history = []
        # - lr_history = []
        # - metric_history = []
        # - patience and metric_higher_is_better
        raise NotImplementedError("Implement __init__")

    def step(self, loss: float, lr: float):
        """Record one training step.

        Args:
            loss: training loss for this step
            lr: learning rate at this step
        """
        # TODO: Increment current_step, append to histories
        raise NotImplementedError("Implement step")

    def next_epoch(self):
        """Increment epoch counter."""
        # TODO: Increment current_epoch
        raise NotImplementedError("Implement next_epoch")

    def update_metric(self, metric: float):
        """Record evaluation metric and update early stopping state.

        Args:
            metric: evaluation metric value
        """
        # TODO: Append to metric_history
        # Check if this is the best metric
        # If improved: update best_metric, best_metric_step, reset steps_without_improvement
        # If not: increment steps_without_improvement
        raise NotImplementedError("Implement update_metric")

    def should_stop(self) -> bool:
        """Check if early stopping criterion is met.

        Returns:
            True if steps_without_improvement >= patience
        """
        # TODO: Implement early stopping check
        raise NotImplementedError("Implement should_stop")

    def state_dict(self) -> dict:
        """Serialize state for checkpointing."""
        # TODO: Return dict with all state variables
        raise NotImplementedError("Implement state_dict")

    def load_state_dict(self, d: dict):
        """Restore state from checkpoint."""
        # TODO: Restore all state variables from dict
        raise NotImplementedError("Implement load_state_dict")

    def get_summary(self) -> dict:
        """Return training summary.

        Returns:
            dict with keys: 'total_steps', 'total_epochs', 'best_metric',
            'best_metric_step', 'final_loss', 'avg_loss_last_10'
        """
        # TODO: Compute and return summary statistics
        raise NotImplementedError("Implement get_summary")
