"""
Exercise 4: Exact Match Metric

Exact Match (EM) is a core evaluation metric for QA tasks. The predicted answer
must exactly match one of the acceptable ground truth answers after normalization.

Normalization typically includes:
- Lowercasing
- Stripping whitespace
- Removing articles (a, an, the)
- Removing punctuation

Reference: slime rm_hub evaluation patterns and standard NLP eval (SQuAD-style EM).

Your task: implement normalize_answer(), exact_match_single(), and
compute_exact_match().

Difficulty: Easy
Framework: numpy / stdlib
"""

import re
import string


def normalize_answer(text: str) -> str:
    """Normalize an answer string for exact match comparison.

    Apply these steps in order:
    1. Lowercase the text
    2. Remove punctuation (all characters in string.punctuation)
    3. Remove articles: "a", "an", "the" (as whole words only)
    4. Collapse multiple whitespace into a single space
    5. Strip leading/trailing whitespace

    Args:
        text: Raw answer string.

    Returns:
        Normalized string.

    Examples:
        >>> normalize_answer("The quick Brown Fox!")
        'quick brown fox'
        >>> normalize_answer("  A  dog's  life  ")
        'dogs life'
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement normalize_answer")


def exact_match_single(prediction: str, ground_truths: list[str]) -> bool:
    """Check if a prediction exactly matches any of the acceptable answers.

    Args:
        prediction: The model's predicted answer string.
        ground_truths: List of acceptable answer strings (at least one).

    Returns:
        True if normalize_answer(prediction) equals normalize_answer(gt)
        for any gt in ground_truths.
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement exact_match_single")


def compute_exact_match(
    predictions: list[str],
    ground_truths_list: list[list[str]],
) -> float:
    """Compute exact match score across a dataset.

    Args:
        predictions: List of N predicted answer strings.
        ground_truths_list: List of N lists, each containing acceptable answers.

    Returns:
        EM score as a float in [0, 1] (fraction of exact matches).
        If predictions is empty, return 0.0.
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement compute_exact_match")
