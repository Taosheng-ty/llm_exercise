"""
Exercise 4: Outcome Reward Model (Rule-Based) - Solution
"""

import re


def extract_from_boxed(response: str) -> str | None:
    """Extract answer from \\boxed{...} in the response."""
    # Find the last \boxed occurrence
    idx = response.rfind("\\boxed")
    if idx < 0:
        return None

    # Find the opening brace
    i = idx + len("\\boxed")
    if i >= len(response) or response[i] != "{":
        return None

    # Match braces to handle nesting
    depth = 0
    start = i
    while i < len(response):
        if response[i] == "{":
            depth += 1
        elif response[i] == "}":
            depth -= 1
            if depth == 0:
                return response[start + 1 : i]
        i += 1

    return None


def extract_from_answer_phrase(response: str) -> str | None:
    """Extract answer from 'the answer is X' pattern."""
    # Find all occurrences, take the last one
    matches = list(re.finditer(r"the answer is\s*", response, re.IGNORECASE))
    if not matches:
        return None

    last_match = matches[-1]
    rest = response[last_match.end():]
    # Capture until newline or end of string, then strip trailing period
    m = re.match(r"([^\n]+)", rest)
    if m:
        result = m.group(1).strip().rstrip(".")
        return result if result else None
    return None


def extract_from_last_line(response: str) -> str | None:
    """Extract the last non-empty line of the response as the answer."""
    lines = response.strip().split("\n")
    for line in reversed(lines):
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def normalize_for_comparison(text: str) -> str:
    """Simple normalization: lowercase, strip whitespace, remove $ and trailing periods."""
    s = text.strip().lower()
    s = s.replace("$", "")
    s = s.rstrip(".")
    s = s.strip()
    return s


def compute_outcome_reward(expected_answer: str, response: str) -> float:
    """Compute reward by extracting answer from response and comparing to expected."""
    strategies = [
        extract_from_boxed,
        extract_from_answer_phrase,
        extract_from_last_line,
    ]

    for strategy in strategies:
        extracted = strategy(response)
        if extracted is not None:
            if normalize_for_comparison(extracted) == normalize_for_comparison(expected_answer):
                return 1.0
            return 0.0

    return 0.0
