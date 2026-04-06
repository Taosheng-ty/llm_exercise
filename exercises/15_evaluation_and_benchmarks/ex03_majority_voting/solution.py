"""
Solution for Exercise 3: Majority Voting for Self-Consistency
"""

import torch
from collections import Counter


def majority_vote(answers: list[str]) -> str | None:
    """Determine the majority answer from a list of responses."""
    valid = [a for a in answers if a]
    if not valid:
        return None

    # Count occurrences, preserving first-occurrence order for ties
    counts: dict[str, int] = {}
    for a in valid:
        counts[a] = counts.get(a, 0) + 1

    max_count = max(counts.values())
    # Return the first answer that has the max count (preserves input order)
    for a in valid:
        if counts[a] == max_count:
            return a
    return None


def weighted_majority_vote(
    answers: list[str],
    confidences: torch.Tensor,
) -> str | None:
    """Determine the majority answer using confidence-weighted voting."""
    weights: dict[str, float] = {}
    first_seen: dict[str, int] = {}

    for i, a in enumerate(answers):
        if not a:
            continue
        w = confidences[i].item()
        weights[a] = weights.get(a, 0.0) + w
        if a not in first_seen:
            first_seen[a] = i

    if not weights:
        return None

    max_weight = max(weights.values())
    # Tie-break by first occurrence
    best = None
    best_idx = float("inf")
    for a, w in weights.items():
        if abs(w - max_weight) < 1e-12 and first_seen[a] < best_idx:
            best = a
            best_idx = first_seen[a]
        elif w > max_weight + 1e-12:
            # shouldn't happen since max_weight is the max
            pass

    # Simpler: just pick the one with max weight, tie-break by first seen
    candidates = [(a, w, first_seen[a]) for a, w in weights.items()]
    candidates.sort(key=lambda x: (-x[1], x[2]))
    return candidates[0][0]


def compute_agreement_rate(
    all_answers: list[list[str]],
    threshold: float = 0.5,
) -> float:
    """Compute the fraction of questions where a clear majority exists."""
    if not all_answers:
        return 0.0

    majority_count = 0
    for answers in all_answers:
        valid = [a for a in answers if a]
        if not valid:
            continue
        counter = Counter(valid)
        most_common_count = counter.most_common(1)[0][1]
        if most_common_count / len(valid) > threshold:
            majority_count += 1

    return majority_count / len(all_answers)
