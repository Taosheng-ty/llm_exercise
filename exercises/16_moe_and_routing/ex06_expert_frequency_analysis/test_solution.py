import importlib.util
import os

import numpy as np

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

expert_utilization_rate = _mod.expert_utilization_rate
load_imbalance_score = _mod.load_imbalance_score
find_dead_experts = _mod.find_dead_experts


def test_utilization_uniform():
    """Perfectly uniform routing should give equal utilization."""
    num_tokens, num_experts = 100, 4
    # Each token uses exactly one expert, cycling through all
    decisions = np.arange(num_tokens).reshape(-1, 1) % num_experts
    util = expert_utilization_rate(decisions, num_experts)

    assert util.shape == (num_experts,)
    np.testing.assert_allclose(util, 0.25, atol=0.01)


def test_utilization_skewed():
    """All tokens to one expert -> utilization = 1 for that expert, 0 for others."""
    decisions = np.zeros((50, 1), dtype=int)
    util = expert_utilization_rate(decisions, num_experts=4)

    assert util[0] == 1.0
    assert util[1] == 0.0
    assert util[2] == 0.0
    assert util[3] == 0.0


def test_utilization_top_k():
    """With top_k=2, a token can use 2 experts."""
    # 4 tokens, each using experts (0,1)
    decisions = np.array([[0, 1], [0, 1], [0, 1], [0, 1]])
    util = expert_utilization_rate(decisions, num_experts=4)

    assert util[0] == 1.0  # All tokens use expert 0
    assert util[1] == 1.0  # All tokens use expert 1
    assert util[2] == 0.0
    assert util[3] == 0.0


def test_load_imbalance_uniform():
    """Perfectly uniform -> imbalance close to 0."""
    decisions = np.arange(200).reshape(-1, 1) % 8
    score = load_imbalance_score(decisions, num_experts=8)
    assert score < 0.05, f"Uniform routing should have low imbalance, got {score}"


def test_load_imbalance_skewed():
    """Highly skewed routing -> high imbalance score."""
    decisions = np.zeros((100, 1), dtype=int)  # All to expert 0
    score = load_imbalance_score(decisions, num_experts=8)
    assert score > 1.0, f"Skewed routing should have high imbalance, got {score}"


def test_find_dead_no_dead():
    """Uniform routing -> no dead experts."""
    decisions = np.arange(800).reshape(-1, 1) % 8
    dead = find_dead_experts(decisions, num_experts=8, threshold=0.01)
    assert dead == []


def test_find_dead_some_dead():
    """Experts 2 and 3 never used -> they are dead."""
    # Only experts 0 and 1 are used
    decisions = np.array([[0, 1]] * 100)
    dead = find_dead_experts(decisions, num_experts=4, threshold=0.01)
    assert dead == [2, 3]


def test_find_dead_with_threshold():
    """Expert used by 1 out of 200 tokens -> utilization 0.005, dead at threshold 0.01."""
    decisions = np.zeros((200, 1), dtype=int)  # All to expert 0
    decisions[0, 0] = 3  # One token uses expert 3

    dead = find_dead_experts(decisions, num_experts=4, threshold=0.01)
    # Expert 1 and 2: utilization = 0, dead
    # Expert 3: utilization = 1/200 = 0.005, dead
    # Expert 0: utilization = 199/200, alive
    assert 1 in dead
    assert 2 in dead
    assert 3 in dead
    assert 0 not in dead


def test_dead_experts_sorted():
    decisions = np.zeros((100, 1), dtype=int)
    dead = find_dead_experts(decisions, num_experts=8, threshold=0.01)
    assert dead == sorted(dead)
