"""Tests for Exercise 05: Win Rate and ELO Rating Computation"""

import importlib.util
import os
import numpy as np
import pytest

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
compute_win_rates = _mod.compute_win_rates
compute_elo_ratings = _mod.compute_elo_ratings


class TestComputeWinRates:
    def test_basic_two_models(self):
        """Simple 2-model comparison."""
        # Model 0 beat model 1 three times, model 1 beat model 0 once
        comp = np.array([
            [0, 3],
            [1, 0],
        ])
        wr = compute_win_rates(comp)
        assert wr[0, 1] == pytest.approx(0.75)
        assert wr[1, 0] == pytest.approx(0.25)

    def test_diagonal_is_half(self):
        """Diagonal should always be 0.5."""
        comp = np.array([
            [0, 5, 2],
            [3, 0, 7],
            [1, 4, 0],
        ])
        wr = compute_win_rates(comp)
        for i in range(3):
            assert wr[i, i] == pytest.approx(0.5)

    def test_no_matches_returns_half(self):
        """If two models never played, win rate should be 0.5."""
        comp = np.array([
            [0, 5, 0],
            [3, 0, 0],
            [0, 0, 0],
        ])
        wr = compute_win_rates(comp)
        # Model 0 vs 2: no matches -> 0.5
        assert wr[0, 2] == pytest.approx(0.5)
        assert wr[2, 0] == pytest.approx(0.5)

    def test_perfect_domination(self):
        """Model that always wins should have 1.0 win rate."""
        comp = np.array([
            [0, 10],
            [0, 0],
        ])
        wr = compute_win_rates(comp)
        assert wr[0, 1] == pytest.approx(1.0)
        assert wr[1, 0] == pytest.approx(0.0)

    def test_output_shape(self):
        """Output should have same shape as input."""
        N = 5
        comp = np.random.randint(0, 10, size=(N, N))
        np.fill_diagonal(comp, 0)
        wr = compute_win_rates(comp)
        assert wr.shape == (N, N)

    def test_win_rates_sum_to_one(self):
        """wr[i][j] + wr[j][i] should be 1.0 for all i != j (when matches exist)."""
        comp = np.array([
            [0, 7, 3],
            [5, 0, 8],
            [2, 6, 0],
        ])
        wr = compute_win_rates(comp)
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert wr[i, j] + wr[j, i] == pytest.approx(1.0)


class TestComputeEloRatings:
    def test_winner_gains_loser_loses(self):
        """Winner should gain rating, loser should lose."""
        matches = [("A", "B", 1.0)]
        ratings = compute_elo_ratings(matches, k=32, initial=1000)
        assert ratings["A"] > 1000
        assert ratings["B"] < 1000

    def test_draw_equal_ratings_no_change(self):
        """Draw between equal-rated players should not change ratings."""
        matches = [("A", "B", 0.5)]
        ratings = compute_elo_ratings(matches, k=32, initial=1000)
        assert ratings["A"] == pytest.approx(1000, abs=1e-10)
        assert ratings["B"] == pytest.approx(1000, abs=1e-10)

    def test_rating_sum_conserved(self):
        """Total rating points should be conserved (zero-sum updates)."""
        matches = [
            ("A", "B", 1.0),
            ("B", "C", 0.0),
            ("A", "C", 0.5),
            ("C", "A", 1.0),
        ]
        ratings = compute_elo_ratings(matches, k=32, initial=1000)
        total = sum(ratings.values())
        expected_total = len(ratings) * 1000
        assert total == pytest.approx(expected_total, abs=1e-6)

    def test_k_factor_scales_updates(self):
        """Higher K should produce larger rating changes."""
        matches = [("A", "B", 1.0)]
        r_small = compute_elo_ratings(matches, k=16, initial=1000)
        r_large = compute_elo_ratings(matches, k=64, initial=1000)
        # Larger K -> bigger gain for winner
        assert r_large["A"] - 1000 > r_small["A"] - 1000

    def test_upset_gives_bigger_gain(self):
        """Beating a higher-rated opponent should give a bigger rating boost."""
        # First establish different ratings
        setup = [("A", "B", 1.0)] * 10  # A becomes much higher rated
        ratings_after_setup = compute_elo_ratings(setup, k=32, initial=1000)

        # Now B beats A (upset) and C beats D (expected, both equal)
        matches_upset = setup + [("B", "A", 1.0)]
        matches_expected = [("C", "D", 1.0)]

        r_upset = compute_elo_ratings(matches_upset, k=32, initial=1000)
        r_expected = compute_elo_ratings(matches_expected, k=32, initial=1000)

        # B's gain from beating higher-rated A should exceed C's gain from beating equal D
        b_gain = r_upset["B"] - ratings_after_setup["B"]
        c_gain = r_expected["C"] - 1000
        assert b_gain > c_gain

    def test_new_models_get_initial_rating(self):
        """Models appearing for the first time should start at initial rating."""
        ratings = compute_elo_ratings([], k=32, initial=1500)
        assert len(ratings) == 0
        # After one match, both start at 1500
        ratings = compute_elo_ratings([("X", "Y", 1.0)], k=32, initial=1500)
        # X won so should be above 1500, Y below
        assert ratings["X"] > 1500
        assert ratings["Y"] < 1500

    def test_many_matches_ordering(self):
        """After many matches, the strongest model should have highest rating."""
        np.random.seed(42)
        matches = []
        # A beats B 80% of time, B beats C 80% of time
        for _ in range(200):
            matches.append(("A", "B", 1.0 if np.random.rand() < 0.8 else 0.0))
            matches.append(("B", "C", 1.0 if np.random.rand() < 0.8 else 0.0))
        ratings = compute_elo_ratings(matches, k=32, initial=1000)
        assert ratings["A"] > ratings["B"] > ratings["C"]

    def test_expected_score_formula(self):
        """Verify the expected score computation indirectly through rating changes."""
        # Equal ratings -> expected score = 0.5 -> win gives K*(1-0.5) = K/2 gain
        matches = [("A", "B", 1.0)]
        ratings = compute_elo_ratings(matches, k=32, initial=1000)
        assert ratings["A"] == pytest.approx(1016.0, abs=1e-6)
        assert ratings["B"] == pytest.approx(984.0, abs=1e-6)
