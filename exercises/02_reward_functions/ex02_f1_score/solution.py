"""
Exercise 2: Token-level F1 Score - Solution
"""

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """Normalize an answer string for token-level comparison."""
    # 1. Lowercase
    text = s.lower()
    # 2. Remove punctuation
    exclude = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in exclude)
    # 3. Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # 4. Fix whitespace
    text = " ".join(text.split())
    return text


def f1_score(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    """Compute token-level F1 score between prediction and ground truth."""
    ZERO_METRIC = (0.0, 0.0, 0.0)

    if prediction is None:
        return ZERO_METRIC

    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    # Special token handling
    if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return ZERO_METRIC

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return (f1, precision, recall)
