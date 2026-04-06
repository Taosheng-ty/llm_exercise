"""
Exercise 7: Aggregate Benchmark Results

After evaluating a model on multiple benchmark tasks, results must be aggregated
into summary statistics. This includes macro/weighted averages, per-category
breakdowns, and confidence intervals via bootstrap resampling.

Your task: implement macro_average(), weighted_average(),
category_averages(), and bootstrap_confidence_interval().

Difficulty: Easy
Framework: numpy
"""

import numpy as np


def macro_average(scores: dict[str, float]) -> float:
    """Compute the unweighted (macro) average of per-task scores.

    Args:
        scores: Dict mapping task names to their metric scores.
                E.g., {"gpqa": 0.45, "math": 0.72, "ifeval": 0.88}

    Returns:
        Mean of all scores. Return 0.0 if scores is empty.
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement macro_average")


def weighted_average(
    scores: dict[str, float],
    weights: dict[str, float],
) -> float:
    """Compute weighted average of per-task scores.

    Args:
        scores: Dict mapping task names to their metric scores.
        weights: Dict mapping task names to their weights.
                 Tasks in scores but not in weights get weight 1.0.
                 Tasks in weights but not in scores are ignored.

    Returns:
        Weighted average: sum(score_i * weight_i) / sum(weight_i).
        Return 0.0 if scores is empty.
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement weighted_average")


def category_averages(
    scores: dict[str, float],
    task_categories: dict[str, str],
) -> dict[str, float]:
    """Compute average score per category.

    Args:
        scores: Dict mapping task names to scores.
        task_categories: Dict mapping task names to category names.
                         E.g., {"gpqa": "reasoning", "math": "reasoning",
                                "ifeval": "instruction_following"}
                         Tasks not in task_categories go into "other".

    Returns:
        Dict mapping category names to average scores for that category.
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement category_averages")


def bootstrap_confidence_interval(
    per_sample_scores: list[float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """Compute bootstrap confidence interval for the mean score.

    Steps:
    1. Set the random seed using np.random.default_rng(seed).
    2. Draw n_bootstrap resamples (with replacement) from per_sample_scores.
    3. Compute the mean of each resample.
    4. Return the mean of means, and the lower/upper bounds at the given
       confidence level (using percentiles).

    Args:
        per_sample_scores: List of individual sample scores.
        n_bootstrap: Number of bootstrap resamples.
        confidence_level: E.g., 0.95 for a 95% confidence interval.
        seed: Random seed for reproducibility.

    Returns:
        Dict with:
        - "mean": float, mean of bootstrap means
        - "lower": float, lower bound of confidence interval
        - "upper": float, upper bound of confidence interval
        - "std": float, standard deviation of bootstrap means

    If per_sample_scores is empty, return mean=0, lower=0, upper=0, std=0.
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement bootstrap_confidence_interval")
