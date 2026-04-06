"""
Exercise 4: Outcome Reward Model (Rule-Based)

In slime's rm_hub, reward models try multiple strategies to extract an answer
from a model response and compare it to the expected answer. This exercise
implements a simplified version of that pattern.

Your task: build a rule-based outcome reward model that:
1. Tries multiple extraction strategies to find the answer in the response
2. Normalizes and compares the extracted answer to the expected answer
3. Returns a reward in [0, 1]

Extraction strategies (try in this order, return first successful match):
  a) \\boxed{...} - LaTeX boxed answer (handle nested braces)
  b) "The answer is X" or "the answer is X" (case-insensitive, X is captured until newline or end of string)
  c) Last line of the response (strip whitespace)

Reference: inspired by slime rm_hub pattern (math_utils.py extract_boxed_answer,
           __init__.py where multiple rm_types dispatch different strategies)

Difficulty: Medium
"""

import re


def extract_from_boxed(response: str) -> str | None:
    """Extract answer from \\boxed{...} in the response.

    Find the LAST occurrence of \\boxed{...} and return its contents.
    Must handle nested braces, e.g., \\boxed{\\frac{1}{2}} -> \\frac{1}{2}

    Args:
        response: Full model response text

    Returns:
        The content inside \\boxed{}, or None if not found

    Examples:
        >>> extract_from_boxed("So \\boxed{42}")
        '42'
        >>> extract_from_boxed("Thus \\boxed{\\frac{1}{2}}")
        '\\frac{1}{2}'
        >>> extract_from_boxed("No boxed here")
        None
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement extract_from_boxed")


def extract_from_answer_phrase(response: str) -> str | None:
    """Extract answer from 'the answer is X' pattern.

    Look for the LAST occurrence of "the answer is" (case-insensitive),
    then capture everything after it until a newline or end of string.
    Strip the captured text (including any trailing period).

    Args:
        response: Full model response text

    Returns:
        Extracted answer string, or None if pattern not found

    Examples:
        >>> extract_from_answer_phrase("Computing... The answer is 42.")
        '42'
        >>> extract_from_answer_phrase("So the answer is 7")
        '7'
        >>> extract_from_answer_phrase("The answer is 3.14")
        '3.14'
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement extract_from_answer_phrase")


def extract_from_last_line(response: str) -> str | None:
    """Extract the last non-empty line of the response as the answer.

    Args:
        response: Full model response text

    Returns:
        Last non-empty line stripped of whitespace, or None if response is empty

    Examples:
        >>> extract_from_last_line("Line 1\\nLine 2\\n42")
        '42'
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement extract_from_last_line")


def normalize_for_comparison(text: str) -> str:
    """Simple normalization: lowercase, strip whitespace, remove $ and trailing periods.

    Args:
        text: Raw text to normalize

    Returns:
        Normalized string
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement normalize_for_comparison")


def compute_outcome_reward(expected_answer: str, response: str) -> float:
    """Compute reward by extracting answer from response and comparing to expected.

    Try extraction strategies in order: boxed -> answer phrase -> last line.
    Use the FIRST strategy that returns a non-None result.
    Compare extracted answer to expected answer using normalize_for_comparison.

    Returns:
        1.0 if extracted answer matches expected, 0.0 otherwise.
        If no extraction strategy succeeds, return 0.0.
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement compute_outcome_reward")
