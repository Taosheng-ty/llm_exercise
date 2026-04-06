"""
Exercise 3: Reward Shaping

In RLHF, we often augment a raw correctness reward with additional shaping
signals to encourage desirable behaviors like conciseness and structured
reasoning.

Given a raw binary reward (1.0 for correct, 0.0 for incorrect), apply:
1. Length penalty: discourage overly verbose responses
2. Format bonus: reward structured reasoning (step markers, conclusion words)

This is a common pattern in reward design for math/reasoning RL training.

Difficulty: Easy
"""


def compute_length_penalty(response: str, target_length: int, alpha: float = 0.5) -> float:
    """Compute a length penalty for responses exceeding target length.

    penalty = -alpha * max(0, len(response) - target_length) / target_length

    The penalty is always <= 0 (no bonus for short responses, only penalty for long ones).

    Args:
        response: The model's response string
        target_length: Desired maximum character count
        alpha: Penalty scaling factor (default 0.5)

    Returns:
        A float <= 0 representing the length penalty

    Examples:
        >>> compute_length_penalty("short", 100)
        0.0
        >>> compute_length_penalty("x" * 200, 100, alpha=0.5)
        -0.5
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement compute_length_penalty")


def compute_format_bonus(response: str, beta: float = 0.1) -> float:
    """Compute a format bonus for well-structured reasoning.

    Award +beta if the response contains at least ONE reasoning marker:
      - "Step 1" (case-insensitive)
      - "Therefore" (case-insensitive)
      - "First," (case-insensitive, must include the comma)
      - "Thus," (case-insensitive, must include the comma)
      - "In conclusion" (case-insensitive)

    Args:
        response: The model's response string
        beta: Bonus value (default 0.1)

    Returns:
        beta if a reasoning marker is found, else 0.0

    Examples:
        >>> compute_format_bonus("Step 1: add numbers. Therefore, 5.")
        0.1
        >>> compute_format_bonus("The answer is 5.")
        0.0
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement compute_format_bonus")


def shape_reward(
    raw_reward: float,
    response: str,
    target_length: int = 500,
    alpha: float = 0.5,
    beta: float = 0.1,
) -> float:
    """Combine raw reward with length penalty and format bonus.

    shaped_reward = raw_reward + length_penalty + format_bonus

    The result is NOT clipped -- it can go below 0 or above 1.

    Args:
        raw_reward: Binary reward (0.0 or 1.0)
        response: Model's response text
        target_length: Target max character count for length penalty
        alpha: Length penalty coefficient
        beta: Format bonus value

    Returns:
        Shaped reward as a float
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement shape_reward")
