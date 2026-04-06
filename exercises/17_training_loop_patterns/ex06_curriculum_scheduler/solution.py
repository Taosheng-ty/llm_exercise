"""
Solution for Exercise 06: Curriculum Learning Scheduler
"""

import numpy as np


class CurriculumScheduler:
    """Schedules training data based on difficulty curriculum."""

    def __init__(self, difficulties: np.ndarray, total_steps: int):
        self.difficulties = difficulties
        self.total_steps = total_steps

    def get_competence(self, step: int, strategy: str) -> float:
        ratio = step / self.total_steps if self.total_steps > 0 else 1.0
        if strategy == "linear":
            return min(1.0, ratio)
        elif strategy == "exponential":
            return min(1.0, 1.0 - np.exp(-5.0 * ratio))
        elif strategy == "competence":
            return min(1.0, np.sqrt(ratio))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def get_available_indices(self, step: int, strategy: str = "linear") -> np.ndarray:
        c = self.get_competence(step, strategy)
        return np.where(self.difficulties <= c)[0]
