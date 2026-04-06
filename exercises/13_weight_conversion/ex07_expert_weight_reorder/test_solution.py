import importlib.util
import os

import numpy as np

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

compute_expert_order = _mod.compute_expert_order
reorder_expert_weights = _mod.reorder_expert_weights
reorder_router_weights = _mod.reorder_router_weights
reorder_moe_layer = _mod.reorder_moe_layer


class TestComputeExpertOrder:
    def test_basic_ordering(self):
        # Expert 2 has highest score, then 0, then 1
        scores = np.array([
            [0.1, 0.0, 0.9],
            [0.3, 0.1, 0.6],
            [0.2, 0.0, 0.8],
        ])
        order = compute_expert_order(scores)
        assert order[0] == 2  # highest total
        assert order[2] == 1  # lowest total

    def test_returns_permutation(self):
        scores = np.random.rand(100, 8)
        order = compute_expert_order(scores)
        assert set(order) == set(range(8))
        assert len(order) == 8

    def test_single_expert(self):
        scores = np.array([[1.0]])
        order = compute_expert_order(scores)
        assert order[0] == 0

    def test_equal_scores(self):
        scores = np.ones((10, 4))
        order = compute_expert_order(scores)
        assert set(order) == {0, 1, 2, 3}


class TestReorderExpertWeights:
    def test_basic_reorder(self):
        weights = {
            0: np.array([1.0, 0.0]),
            1: np.array([0.0, 1.0]),
            2: np.array([2.0, 2.0]),
        }
        # Reverse order: expert 2 -> pos 0, expert 1 -> pos 1, expert 0 -> pos 2
        new_order = np.array([2, 1, 0])
        reordered = reorder_expert_weights(weights, new_order)

        assert np.array_equal(reordered[0], weights[2])
        assert np.array_equal(reordered[1], weights[1])
        assert np.array_equal(reordered[2], weights[0])

    def test_identity_reorder(self):
        weights = {i: np.random.randn(4, 4) for i in range(4)}
        identity = np.array([0, 1, 2, 3])
        reordered = reorder_expert_weights(weights, identity)
        for i in range(4):
            assert np.array_equal(reordered[i], weights[i])

    def test_keys_are_new_indices(self):
        weights = {0: np.zeros(2), 1: np.ones(2)}
        reordered = reorder_expert_weights(weights, np.array([1, 0]))
        assert set(reordered.keys()) == {0, 1}


class TestReorderRouterWeights:
    def test_basic(self):
        router = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 2.0],
        ])
        new_order = np.array([2, 0, 1])
        reordered = reorder_router_weights(router, new_order)

        assert np.array_equal(reordered[0], router[2])
        assert np.array_equal(reordered[1], router[0])
        assert np.array_equal(reordered[2], router[1])

    def test_shape_preserved(self):
        router = np.random.randn(8, 64)
        order = np.array([7, 6, 5, 4, 3, 2, 1, 0])
        reordered = reorder_router_weights(router, order)
        assert reordered.shape == router.shape


class TestReorderMoeLayer:
    def test_end_to_end(self):
        num_experts = 4
        hidden_dim = 8

        # Expert 3 has highest routing score, then 1, then 0, then 2
        np.random.seed(42)
        routing_scores = np.array([
            [0.1, 0.3, 0.0, 0.6],
            [0.1, 0.2, 0.1, 0.6],
            [0.2, 0.3, 0.0, 0.5],
        ])

        expert_weights = {i: np.random.randn(16, hidden_dim) for i in range(num_experts)}
        router_weight = np.random.randn(num_experts, hidden_dim)

        reordered_experts, reordered_router, new_order = reorder_moe_layer(
            expert_weights, router_weight, routing_scores
        )

        # Expert 3 should be first (highest total score)
        assert new_order[0] == 3

        # Verify expert weights moved correctly
        for new_idx, old_idx in enumerate(new_order):
            assert np.array_equal(reordered_experts[new_idx], expert_weights[old_idx])

        # Verify router weights moved correctly
        for new_idx, old_idx in enumerate(new_order):
            assert np.array_equal(reordered_router[new_idx], router_weight[old_idx])

    def test_routing_still_works(self):
        """After reordering, applying the router to a hidden state should
        select the same expert (just with a different index)."""
        num_experts = 4
        hidden_dim = 16
        np.random.seed(123)

        routing_scores = np.random.rand(50, num_experts)
        expert_weights = {i: np.random.randn(8, hidden_dim) for i in range(num_experts)}
        router_weight = np.random.randn(num_experts, hidden_dim)

        reordered_experts, reordered_router, new_order = reorder_moe_layer(
            expert_weights, router_weight, routing_scores
        )

        # For a test hidden state, the top-1 expert output should be the same
        h = np.random.randn(hidden_dim)
        old_scores = router_weight @ h
        old_top = np.argmax(old_scores)

        new_scores = reordered_router @ h
        new_top = np.argmax(new_scores)

        # The new top expert should map back to the same old expert
        assert new_order[new_top] == old_top
