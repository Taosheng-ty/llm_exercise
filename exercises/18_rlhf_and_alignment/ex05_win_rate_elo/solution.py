"""
Solution for Exercise 05: Win Rate and ELO Rating Computation
"""

import numpy as np


def compute_win_rates(comparison_matrix: np.ndarray) -> np.ndarray:
    """Compute pairwise win rate matrix from comparison counts.

    win_rate[i][j] = wins_ij / (wins_ij + wins_ji), with 0.5 for ties or no data.
    """
    N = comparison_matrix.shape[0]
    total = comparison_matrix + comparison_matrix.T
    win_rate = np.full((N, N), 0.5)

    nonzero = total > 0
    win_rate[nonzero] = comparison_matrix[nonzero] / total[nonzero]

    # Diagonal is 0.5 by convention
    np.fill_diagonal(win_rate, 0.5)

    return win_rate


def compute_elo_ratings(
    matches: list[tuple[str, str, float]],
    k: float = 32,
    initial: float = 1000,
) -> dict[str, float]:
    """Compute ELO ratings from a sequence of match results.

    For each match:
        E_A = 1 / (1 + 10^((R_B - R_A) / 400))
        R_A' = R_A + K * (S_A - E_A)
        R_B' = R_B + K * (S_B - E_B)
    """
    ratings: dict[str, float] = {}

    for model_a, model_b, score_a in matches:
        # Initialize new models
        if model_a not in ratings:
            ratings[model_a] = initial
        if model_b not in ratings:
            ratings[model_b] = initial

        r_a = ratings[model_a]
        r_b = ratings[model_b]

        # Expected scores
        e_a = 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))
        e_b = 1.0 - e_a

        # Actual scores
        score_b = 1.0 - score_a

        # Update ratings
        ratings[model_a] = r_a + k * (score_a - e_a)
        ratings[model_b] = r_b + k * (score_b - e_b)

    return ratings
