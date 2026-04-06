"""
Exercise 06: Curriculum Learning Scheduler

Difficulty: Medium
Framework: numpy

Background:
    Curriculum learning presents training data in order of increasing
    difficulty, which can improve convergence. Common strategies:

    1. Linear curriculum: difficulty threshold increases linearly with step
    2. Exponential curriculum: difficulty threshold increases exponentially
    3. Competence-based: at step t, model competence c(t) determines which
       data (difficulty <= c(t)) is available

    Reference: In slime's rollout pipeline, data ordering and filtering
    can be thought of as a form of curriculum.

Implement the CurriculumScheduler class:
    - __init__(difficulties, total_steps): difficulties is array of per-sample difficulty
    - get_available_indices(step, strategy): returns indices of samples available at this step
    - Strategies: 'linear', 'exponential', 'competence'

Difficulty values are in [0, 1] where 0 = easiest, 1 = hardest.

For each strategy, compute a competence threshold c(step):
    - linear: c = min(1.0, step / total_steps)
    - exponential: c = min(1.0, 1 - exp(-5 * step / total_steps))
    - competence: c = min(1.0, sqrt(step / total_steps))

Return indices where difficulty[i] <= c.
"""

import numpy as np


class CurriculumScheduler:
    """Schedules training data based on difficulty curriculum."""

    def __init__(self, difficulties: np.ndarray, total_steps: int):
        """
        Args:
            difficulties: (N,) array of difficulty scores in [0, 1]
            total_steps: total number of training steps
        """
        self.difficulties = difficulties
        self.total_steps = total_steps

    def get_competence(self, step: int, strategy: str) -> float:
        """Compute model competence at given step.

        Args:
            step: current training step
            strategy: one of 'linear', 'exponential', 'competence'

        Returns:
            Competence threshold in [0, 1].
        """
        # TODO: Implement competence computation for each strategy
        # Hint: See formulas in module docstring
        raise NotImplementedError("Implement get_competence")

    def get_available_indices(self, step: int, strategy: str = "linear") -> np.ndarray:
        """Get indices of samples available at this training step.

        Args:
            step: current training step
            strategy: curriculum strategy name

        Returns:
            1D array of integer indices where difficulty <= competence.
        """
        # TODO: Compute competence, then return indices where difficulty <= competence
        raise NotImplementedError("Implement get_available_indices")
