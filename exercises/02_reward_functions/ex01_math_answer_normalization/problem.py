"""
Exercise 1: Math Answer Normalization

In RL-based math training (e.g., DeepScaler, DAPO), we need to compare a model's
answer to the ground truth. But answers can appear in many equivalent forms:
  - "\\frac{1}{2}" == "1/2" == "0.5"
  - "\\sqrt{3}" == "sqrt(3)"
  - "$42$" == "42"
  - "x = 7" == "7"

Your task: implement normalize_math_answer() that strips LaTeX formatting and
normalizes math expressions so equivalent answers produce the same string.

Reference: slime/rollout/rm_hub/math_utils.py  _strip_string()

Difficulty: Medium
"""

import re


def normalize_math_answer(answer: str) -> str:
    """Normalize a math answer string for comparison.

    Normalization steps (apply in this order):
    1. Strip leading/trailing whitespace and remove newlines
    2. Remove dollar signs ('$')
    3. Remove LaTeX commands: \\left, \\right, \\text{ ...} (unit removal)
    4. Replace \\tfrac and \\dfrac with \\frac
    5. Remove degree notation: ^{\\circ} and ^\\circ
    6. Remove percentage signs (\\% and literal %)
    7. Handle leading decimals: ".5" -> "0.5"
    8. Handle "variable = value" patterns: if the string has exactly one '='
       and the left side is 1-2 characters, keep only the right side
    9. Convert \\frac{a}{b} -> a/b  (simple replacement)
    10. Convert \\sqrt{x} -> sqrt(x)
    11. Remove all remaining spaces
    12. Lowercase the result

    Args:
        answer: Raw math answer string, possibly with LaTeX formatting

    Returns:
        Normalized string suitable for comparison

    Examples:
        >>> normalize_math_answer("\\frac{1}{2}")
        '1/2'
        >>> normalize_math_answer("\\sqrt{3}")
        'sqrt(3)'
        >>> normalize_math_answer("$42$")
        '42'
        >>> normalize_math_answer("x = 7")
        '7'
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement normalize_math_answer")


def answers_are_equivalent(answer1: str, answer2: str) -> bool:
    """Check if two math answers are equivalent after normalization.

    Additionally, handle the special case where one normalized form is a
    simple fraction "a/b" (both a and b are integers, b != 0) and the other
    is a decimal -- compare their float values with tolerance 1e-9.

    Args:
        answer1: First math answer
        answer2: Second math answer

    Returns:
        True if the answers are considered equivalent

    Examples:
        >>> answers_are_equivalent("\\frac{1}{2}", "0.5")
        True
        >>> answers_are_equivalent("3", "3.0")
        True  # because "3.0" normalizes and float comparison works
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement answers_are_equivalent")
