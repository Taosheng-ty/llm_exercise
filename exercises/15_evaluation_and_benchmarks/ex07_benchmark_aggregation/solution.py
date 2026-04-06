"""
Solution for Exercise 7: Aggregate Benchmark Results
"""

import numpy as np


def macro_average(scores: dict[str, float]) -> float:
    """Compute the unweighted (macro) average of per-task scores."""
    if not scores:
        return 0.0
    return float(np.mean(list(scores.values())))


def weighted_average(
    scores: dict[str, float],
    weights: dict[str, float],
) -> float:
    """Compute weighted average of per-task scores."""
    if not scores:
        return 0.0
    total_weight = 0.0
    total_score = 0.0
    for task, score in scores.items():
        w = weights.get(task, 1.0)
        total_score += score * w
        total_weight += w
    if total_weight == 0:
        return 0.0
    return total_score / total_weight


def category_averages(
    scores: dict[str, float],
    task_categories: dict[str, str],
) -> dict[str, float]:
    """Compute average score per category."""
    cat_scores: dict[str, list[float]] = {}
    for task, score in scores.items():
        cat = task_categories.get(task, "other")
        cat_scores.setdefault(cat, []).append(score)

    return {cat: float(np.mean(vals)) for cat, vals in cat_scores.items()}


def bootstrap_confidence_interval(
    per_sample_scores: list[float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """Compute bootstrap confidence interval for the mean score."""
    if not per_sample_scores:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "std": 0.0}

    rng = np.random.default_rng(seed)
    scores = np.array(per_sample_scores)
    n = len(scores)

    bootstrap_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        resample = rng.choice(scores, size=n, replace=True)
        bootstrap_means[i] = resample.mean()

    alpha = 1.0 - confidence_level
    lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))

    return {
        "mean": float(bootstrap_means.mean()),
        "lower": lower,
        "upper": upper,
        "std": float(bootstrap_means.std()),
    }
