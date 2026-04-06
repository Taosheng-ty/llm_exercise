import importlib.util
import os

import numpy as np

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

apply_capacity_factor = _mod.apply_capacity_factor


def test_output_shapes():
    decisions = np.array([[0, 1], [2, 3], [0, 1], [2, 3]])
    weights = np.array([[0.6, 0.4], [0.5, 0.5], [0.7, 0.3], [0.4, 0.6]])

    adj_w, drop_rate = apply_capacity_factor(decisions, weights, num_experts=4, capacity_factor=1.0)

    assert adj_w.shape == weights.shape
    assert isinstance(drop_rate, float)


def test_no_drop_balanced():
    """Perfectly balanced routing with CF=1.0 should drop nothing."""
    # 4 tokens, 4 experts, top_k=1, each expert gets exactly 1 token
    decisions = np.array([[0], [1], [2], [3]])
    weights = np.array([[1.0], [1.0], [1.0], [1.0]])

    adj_w, drop_rate = apply_capacity_factor(decisions, weights, num_experts=4, capacity_factor=1.0)

    np.testing.assert_array_equal(adj_w, weights)
    assert drop_rate == 0.0


def test_drops_excess_tokens():
    """All tokens to one expert with CF=1.0 should drop most."""
    # 4 tokens all to expert 0, capacity = 1.0 * 4/4 = 1
    decisions = np.array([[0], [0], [0], [0]])
    weights = np.array([[1.0], [1.0], [1.0], [1.0]])

    adj_w, drop_rate = apply_capacity_factor(decisions, weights, num_experts=4, capacity_factor=1.0)

    # Only 1 token should remain (capacity = 1)
    assert np.sum(adj_w > 0) == 1
    assert drop_rate == 0.75  # 3 out of 4 dropped


def test_higher_capacity_factor():
    """Higher CF allows more tokens per expert."""
    decisions = np.array([[0], [0], [0], [0]])
    weights = np.array([[1.0], [1.0], [1.0], [1.0]])

    adj_w, drop_rate = apply_capacity_factor(decisions, weights, num_experts=4, capacity_factor=2.0)

    # capacity = 2.0 * 4/4 = 2 tokens per expert
    assert np.sum(adj_w > 0) == 2
    assert drop_rate == 0.5


def test_preserves_kept_weights():
    """Weights for kept tokens should remain unchanged."""
    decisions = np.array([[0, 1], [0, 1]])
    weights = np.array([[0.7, 0.3], [0.6, 0.4]])

    adj_w, _ = apply_capacity_factor(decisions, weights, num_experts=2, capacity_factor=1.0)

    # capacity = 1.0 * (2*2)/2 = 2 per expert, all fit
    np.testing.assert_array_equal(adj_w, weights)


def test_drops_later_tokens_first():
    """Earlier tokens (lower index) should be kept, later dropped."""
    # 4 tokens to expert 0, capacity = 2
    decisions = np.array([[0], [0], [0], [0]])
    weights = np.array([[0.1], [0.2], [0.3], [0.4]])

    adj_w, _ = apply_capacity_factor(decisions, weights, num_experts=2, capacity_factor=1.0)
    # capacity = 1.0 * 4/2 = 2

    # First 2 tokens should be kept
    assert adj_w[0, 0] == 0.1
    assert adj_w[1, 0] == 0.2
    assert adj_w[2, 0] == 0.0
    assert adj_w[3, 0] == 0.0


def test_drop_rate_range():
    """Drop rate should be between 0 and 1."""
    np.random.seed(42)
    decisions = np.random.randint(0, 8, (100, 2))
    weights = np.random.rand(100, 2)

    _, drop_rate = apply_capacity_factor(decisions, weights, num_experts=8, capacity_factor=1.0)

    assert 0.0 <= drop_rate <= 1.0


def test_does_not_modify_input():
    """Input arrays should not be modified."""
    decisions = np.array([[0], [0], [0], [0]])
    weights = np.array([[1.0], [1.0], [1.0], [1.0]])
    weights_copy = weights.copy()

    apply_capacity_factor(decisions, weights, num_experts=4, capacity_factor=1.0)

    np.testing.assert_array_equal(weights, weights_copy)
