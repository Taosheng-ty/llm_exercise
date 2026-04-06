"""
Exercise 2: Token-level F1 Score

In QA evaluation (e.g., HotpotQA, SQuAD), we compute token-level F1 between
the model's prediction and the gold answer. This is used as a reward signal
in RL-based training for open-ended QA tasks.

Steps:
1. Normalize both strings (lowercase, remove articles, remove punctuation, fix whitespace)
2. Tokenize by splitting on whitespace
3. Compute precision = |common tokens| / |prediction tokens|
4. Compute recall    = |common tokens| / |gold tokens|
5. F1 = 2 * precision * recall / (precision + recall)

Reference: slime/rollout/rm_hub/f1.py

Difficulty: Easy
"""

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """Normalize an answer string for token-level comparison.

    Apply these transformations in order:
    1. Lowercase
    2. Remove punctuation (all characters in string.punctuation)
    3. Remove articles ("a", "an", "the") as whole words
    4. Collapse multiple whitespace into single spaces, strip edges

    Args:
        s: Raw answer string

    Returns:
        Normalized string

    Examples:
        >>> normalize_answer("The quick Brown fox!")
        'quick brown fox'
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement normalize_answer")


def f1_score(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    """Compute token-level F1 score between prediction and ground truth.

    Special cases:
    - If prediction is None, return (0, 0, 0)
    - If either normalized string is a special token ("yes", "no", "noanswer")
      and they don't match, return (0, 0, 0)
    - If there are no common tokens, return (0, 0, 0)

    Args:
        prediction: Model's predicted answer (can be None)
        ground_truth: Gold/reference answer

    Returns:
        Tuple of (f1, precision, recall), each in [0, 1]

    Examples:
        >>> f1_score("the cat sat", "a cat sat down")
        (0.8, 1.0, 0.666...)
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement f1_score")
