"""
Exercise 05: Win Rate and ELO Rating Computation

Difficulty: Medium
Framework: numpy

Background:
    When comparing multiple LLMs, pairwise comparisons are often used to
    establish a ranking. Two key tools:

    1. Win Rate Matrix: Given N models compared pairwise, compute the fraction
       of times each model wins against each other model.

    2. ELO Ratings: An iterative rating system where models gain/lose points
       based on match outcomes relative to expectations.

       Expected score: E_A = 1 / (1 + 10^((R_B - R_A) / 400))
       Update: R_A' = R_A + K * (S_A - E_A)

       where S_A is the actual score (1 for win, 0.5 for draw, 0 for loss),
       and K controls the update magnitude.

Implement:
    compute_win_rates(comparison_matrix) -> win_rate_matrix
        Given a matrix where comparison_matrix[i][j] = number of times model i
        beat model j, compute the win rate matrix where win_rate[i][j] =
        fraction of matches between i and j that i won.

    compute_elo_ratings(matches, k=32, initial=1000) -> dict[str, float]
        Given a list of match results, compute ELO ratings for all models.

Args for compute_win_rates:
    comparison_matrix: np.ndarray of shape (N, N), comparison_matrix[i][j]
        is the number of wins of model i over model j. Diagonal is ignored.

Returns:
    win_rate_matrix: np.ndarray of shape (N, N), win_rate[i][j] = wins_i / (wins_i + wins_j).
        Diagonal should be 0.5. If both wins are 0, set to 0.5.

Args for compute_elo_ratings:
    matches: list of tuples (model_a: str, model_b: str, score_a: float)
        where score_a is 1.0 (a wins), 0.0 (b wins), or 0.5 (draw)
    k: float, ELO K-factor (default 32)
    initial: float, starting rating for all models (default 1000)

Returns:
    ratings: dict mapping model name -> final ELO rating
"""

import numpy as np


def compute_win_rates(comparison_matrix: np.ndarray) -> np.ndarray:
    """Compute pairwise win rate matrix from comparison counts.

    Args:
        comparison_matrix: (N, N) where [i][j] = number of times i beat j

    Returns:
        (N, N) win rate matrix where [i][j] = fraction of i's wins over j
    """
    # TODO: Implement win rate computation
    # Hint 1: total_matches[i][j] = comparison_matrix[i][j] + comparison_matrix[j][i]
    # Hint 2: win_rate[i][j] = comparison_matrix[i][j] / total_matches[i][j]
    # Hint 3: Handle division by zero (no matches between i and j) -> 0.5
    # Hint 4: Diagonal should be 0.5
    raise NotImplementedError("Implement compute_win_rates")


def compute_elo_ratings(
    matches: list[tuple[str, str, float]],
    k: float = 32,
    initial: float = 1000,
) -> dict[str, float]:
    """Compute ELO ratings from a sequence of match results.

    Args:
        matches: list of (model_a, model_b, score_a) tuples
        k: K-factor controlling update magnitude
        initial: starting rating for new models

    Returns:
        dict mapping model name to final ELO rating
    """
    # TODO: Implement ELO rating computation
    # Hint 1: Initialize all models with the initial rating
    # Hint 2: For each match, compute expected scores
    # Hint 3: E_A = 1 / (1 + 10**((R_B - R_A) / 400))
    # Hint 4: Update: R_A += K * (S_A - E_A), R_B += K * (S_B - E_B)
    raise NotImplementedError("Implement compute_elo_ratings")
