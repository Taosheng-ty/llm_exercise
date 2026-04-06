"""
Solution for Exercise 07: Training State Manager
"""

import numpy as np


class TrainingStateManager:
    """Manages training state for checkpointing and early stopping."""

    def __init__(self, patience: int = 10, metric_higher_is_better: bool = True):
        self.patience = patience
        self.metric_higher_is_better = metric_higher_is_better
        self.current_step = 0
        self.current_epoch = 0
        self.best_metric = None
        self.best_metric_step = 0
        self.steps_without_improvement = 0
        self.loss_history = []
        self.lr_history = []
        self.metric_history = []

    def step(self, loss: float, lr: float):
        self.current_step += 1
        self.loss_history.append(loss)
        self.lr_history.append(lr)

    def next_epoch(self):
        self.current_epoch += 1

    def update_metric(self, metric: float):
        self.metric_history.append(metric)

        improved = False
        if self.best_metric is None:
            improved = True
        elif self.metric_higher_is_better:
            improved = metric > self.best_metric
        else:
            improved = metric < self.best_metric

        if improved:
            self.best_metric = metric
            self.best_metric_step = self.current_step
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1

    def should_stop(self) -> bool:
        return self.steps_without_improvement >= self.patience

    def state_dict(self) -> dict:
        return {
            "current_step": self.current_step,
            "current_epoch": self.current_epoch,
            "best_metric": self.best_metric,
            "best_metric_step": self.best_metric_step,
            "steps_without_improvement": self.steps_without_improvement,
            "loss_history": list(self.loss_history),
            "lr_history": list(self.lr_history),
            "metric_history": list(self.metric_history),
            "patience": self.patience,
            "metric_higher_is_better": self.metric_higher_is_better,
        }

    def load_state_dict(self, d: dict):
        self.current_step = d["current_step"]
        self.current_epoch = d["current_epoch"]
        self.best_metric = d["best_metric"]
        self.best_metric_step = d["best_metric_step"]
        self.steps_without_improvement = d["steps_without_improvement"]
        self.loss_history = list(d["loss_history"])
        self.lr_history = list(d["lr_history"])
        self.metric_history = list(d["metric_history"])
        self.patience = d["patience"]
        self.metric_higher_is_better = d["metric_higher_is_better"]

    def get_summary(self) -> dict:
        final_loss = self.loss_history[-1] if self.loss_history else None
        recent = self.loss_history[-10:] if self.loss_history else []
        avg_loss_last_10 = float(np.mean(recent)) if recent else None

        return {
            "total_steps": self.current_step,
            "total_epochs": self.current_epoch,
            "best_metric": self.best_metric,
            "best_metric_step": self.best_metric_step,
            "final_loss": final_loss,
            "avg_loss_last_10": avg_loss_last_10,
        }
