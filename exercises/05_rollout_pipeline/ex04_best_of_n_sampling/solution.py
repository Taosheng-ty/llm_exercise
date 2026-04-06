"""
Solution for Exercise 04: Best-of-N Sampling with Rejection
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class Response:
    """A single generated response with its score."""
    text: str
    score: float


@dataclass
class SelectionResult:
    """Result of a selection operation."""
    selected: Optional[Response]
    method: str
    accepted: bool


@dataclass
class BatchSelectionStats:
    """Statistics for a batch of selections."""
    total_prompts: int
    accepted_count: int
    rejected_count: int
    mean_selected_score: float
    acceptance_rate: float


def greedy_best(responses: List[Response]) -> SelectionResult:
    if not responses:
        raise ValueError("responses must not be empty.")
    best = max(responses, key=lambda r: r.score)
    return SelectionResult(selected=best, method="greedy", accepted=True)


def weighted_sample(responses: List[Response], seed: int = 42) -> SelectionResult:
    if not responses:
        raise ValueError("responses must not be empty.")
    rng = np.random.RandomState(seed)
    scores = np.array([r.score for r in responses], dtype=np.float64)
    weights = scores - scores.min() + 1e-6
    probs = weights / weights.sum()
    idx = rng.choice(len(responses), p=probs)
    return SelectionResult(selected=responses[idx], method="weighted", accepted=True)


def rejection_sample(
    responses: List[Response], threshold: float, seed: int = 42
) -> SelectionResult:
    if not responses:
        raise ValueError("responses must not be empty.")
    best = max(responses, key=lambda r: r.score)
    if best.score >= threshold:
        return SelectionResult(selected=best, method="rejection", accepted=True)
    return SelectionResult(selected=None, method="rejection", accepted=False)


def batch_best_of_n(
    prompt_responses: List[List[Response]],
    method: str = "greedy",
    threshold: float = 0.0,
    seed: int = 42,
) -> Tuple[List[SelectionResult], BatchSelectionStats]:
    if method not in ("greedy", "weighted", "rejection"):
        raise ValueError(f"Unknown method: {method}")

    results = []
    for i, responses in enumerate(prompt_responses):
        if method == "greedy":
            result = greedy_best(responses)
        elif method == "weighted":
            result = weighted_sample(responses, seed=seed + i)
        else:
            result = rejection_sample(responses, threshold=threshold, seed=seed + i)
        results.append(result)

    accepted = [r for r in results if r.accepted]
    rejected_count = len(results) - len(accepted)
    scores = [r.selected.score for r in accepted if r.selected is not None]
    mean_score = float(np.mean(scores)) if scores else 0.0

    stats = BatchSelectionStats(
        total_prompts=len(results),
        accepted_count=len(accepted),
        rejected_count=rejected_count,
        mean_selected_score=mean_score,
        acceptance_rate=len(accepted) / len(results) if results else 0.0,
    )
    return results, stats
