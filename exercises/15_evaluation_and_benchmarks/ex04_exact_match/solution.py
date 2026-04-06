"""
Solution for Exercise 4: Exact Match Metric
"""

import re
import string


def normalize_answer(text: str) -> str:
    """Normalize an answer string for exact match comparison."""
    # 1. Lowercase
    text = text.lower()
    # 2. Remove punctuation
    text = "".join(ch for ch in text if ch not in string.punctuation)
    # 3. Remove articles (whole words)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # 4. Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    # 5. Strip
    text = text.strip()
    return text


def exact_match_single(prediction: str, ground_truths: list[str]) -> bool:
    """Check if a prediction exactly matches any of the acceptable answers."""
    norm_pred = normalize_answer(prediction)
    return any(normalize_answer(gt) == norm_pred for gt in ground_truths)


def compute_exact_match(
    predictions: list[str],
    ground_truths_list: list[list[str]],
) -> float:
    """Compute exact match score across a dataset."""
    if not predictions:
        return 0.0
    matches = sum(
        exact_match_single(pred, gts)
        for pred, gts in zip(predictions, ground_truths_list)
    )
    return matches / len(predictions)
