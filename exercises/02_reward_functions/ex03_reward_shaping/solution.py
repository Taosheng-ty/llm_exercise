"""
Exercise 3: Reward Shaping - Solution
"""

import re


def compute_length_penalty(response: str, target_length: int, alpha: float = 0.5) -> float:
    """Compute a length penalty for responses exceeding target length."""
    excess = max(0, len(response) - target_length)
    return -alpha * excess / target_length


def compute_format_bonus(response: str, beta: float = 0.1) -> float:
    """Compute a format bonus for well-structured reasoning."""
    markers = [
        r"step 1",
        r"therefore",
        r"first,",
        r"thus,",
        r"in conclusion",
    ]
    lower_response = response.lower()
    for marker in markers:
        if marker in lower_response:
            return beta
    return 0.0


def shape_reward(
    raw_reward: float,
    response: str,
    target_length: int = 500,
    alpha: float = 0.5,
    beta: float = 0.1,
) -> float:
    """Combine raw reward with length penalty and format bonus."""
    length_penalty = compute_length_penalty(response, target_length, alpha)
    format_bonus = compute_format_bonus(response, beta)
    return raw_reward + length_penalty + format_bonus
