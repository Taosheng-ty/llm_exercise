"""
Exercise 1: Math Answer Normalization - Solution
"""

import re


def normalize_math_answer(answer: str) -> str:
    """Normalize a math answer string for comparison."""
    # 1. Strip whitespace and remove newlines
    s = answer.strip().replace("\n", "")

    # 2. Remove dollar signs
    s = s.replace("$", "")

    # 3. Remove LaTeX commands: \left, \right, \text{ ...} (unit removal)
    s = s.replace("\\left", "")
    s = s.replace("\\right", "")
    # Remove \text{ ...} -- used for units
    if "\\text{ " in s:
        s = s.split("\\text{ ")[0]

    # 4. Replace \tfrac and \dfrac with \frac
    s = s.replace("\\tfrac", "\\frac")
    s = s.replace("\\dfrac", "\\frac")

    # 5. Remove degree notation
    s = s.replace("^{\\circ}", "")
    s = s.replace("^\\circ", "")

    # 6. Remove percentage signs
    s = s.replace("\\%", "")
    s = s.replace("%", "")

    # 7. Handle leading decimals
    s = s.replace(" .", " 0.")
    if s.startswith("."):
        s = "0" + s

    # 8. Handle "variable = value" pattern
    if s.count("=") == 1:
        left, right = s.split("=")
        if len(left.strip()) <= 2:
            s = right.strip()

    # 9. Convert \frac{a}{b} -> a/b
    frac_pattern = re.compile(r"\\frac\{([^}]*)\}\{([^}]*)\}")
    s = frac_pattern.sub(r"\1/\2", s)

    # 10. Convert \sqrt{x} -> sqrt(x)
    sqrt_pattern = re.compile(r"\\sqrt\{([^}]*)\}")
    s = sqrt_pattern.sub(r"sqrt(\1)", s)

    # 11. Remove all remaining spaces
    s = s.replace(" ", "")

    # 12. Lowercase
    s = s.lower()

    return s


def answers_are_equivalent(answer1: str, answer2: str) -> bool:
    """Check if two math answers are equivalent after normalization."""
    norm1 = normalize_math_answer(answer1)
    norm2 = normalize_math_answer(answer2)

    if norm1 == norm2:
        return True

    # Try numeric comparison: handle fraction vs decimal
    def to_float(s):
        try:
            # Try direct float conversion
            return float(s)
        except ValueError:
            pass
        # Try fraction a/b
        match = re.match(r"^(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)$", s)
        if match:
            num, den = float(match.group(1)), float(match.group(2))
            if den != 0:
                return num / den
        return None

    f1 = to_float(norm1)
    f2 = to_float(norm2)
    if f1 is not None and f2 is not None:
        return abs(f1 - f2) < 1e-9

    return False
