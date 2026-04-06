"""Tests for Exercise 01: GPU Placement."""

import importlib.util
import os
import pytest

# Load solution from same directory
_spec = importlib.util.spec_from_file_location(
    "solution_ex01", os.path.join(os.path.dirname(__file__), "solution.py")
)
_sol = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sol)
allocate_gpus = _sol.allocate_gpus


def test_basic_no_colocate():
    """Basic allocation with separate GPUs for each role."""
    result = allocate_gpus(
        total_gpus=8,
        actor_gpu_count=4,
        critic_gpu_count=2,
        rollout_gpu_count=2,
        colocate=False,
        use_critic=True,
    )
    assert result["actor"] == [0, 1, 2, 3]
    assert result["critic"] == [4, 5]
    assert result["rollout"] == [6, 7]


def test_colocate_rollout_shares_actor():
    """Colocate mode: rollout shares actor GPUs."""
    result = allocate_gpus(
        total_gpus=8,
        actor_gpu_count=4,
        critic_gpu_count=2,
        rollout_gpu_count=2,
        colocate=True,
        use_critic=True,
    )
    assert result["actor"] == [0, 1, 2, 3]
    assert result["critic"] == [4, 5]
    # Rollout shares first 2 of actor's GPUs
    assert result["rollout"] == [0, 1]


def test_no_critic():
    """When use_critic=False, critic should be None."""
    result = allocate_gpus(
        total_gpus=8,
        actor_gpu_count=4,
        critic_gpu_count=99,  # Should be ignored
        rollout_gpu_count=2,
        colocate=False,
        use_critic=False,
    )
    assert result["actor"] == [0, 1, 2, 3]
    assert result["critic"] is None
    assert result["rollout"] == [4, 5]


def test_not_enough_gpus_raises():
    """Should raise ValueError when GPUs are insufficient."""
    with pytest.raises(ValueError, match="Not enough GPUs"):
        allocate_gpus(
            total_gpus=4,
            actor_gpu_count=2,
            critic_gpu_count=2,
            rollout_gpu_count=2,
            colocate=False,
            use_critic=True,
        )


def test_colocate_actor_smaller_than_rollout_raises():
    """Colocate should fail if actor has fewer GPUs than rollout needs."""
    with pytest.raises(ValueError, match="Colocate mode requires"):
        allocate_gpus(
            total_gpus=8,
            actor_gpu_count=2,
            critic_gpu_count=2,
            rollout_gpu_count=4,
            colocate=True,
            use_critic=True,
        )


def test_no_overlap_when_not_colocated():
    """Non-colocated roles must have disjoint GPU sets."""
    result = allocate_gpus(
        total_gpus=12,
        actor_gpu_count=4,
        critic_gpu_count=4,
        rollout_gpu_count=4,
        colocate=False,
        use_critic=True,
    )
    actor_set = set(result["actor"])
    critic_set = set(result["critic"])
    rollout_set = set(result["rollout"])
    assert actor_set.isdisjoint(critic_set)
    assert actor_set.isdisjoint(rollout_set)
    assert critic_set.isdisjoint(rollout_set)


def test_exact_gpu_count():
    """Allocation uses exactly the right number of GPUs."""
    result = allocate_gpus(
        total_gpus=6,
        actor_gpu_count=2,
        critic_gpu_count=2,
        rollout_gpu_count=2,
        colocate=False,
        use_critic=True,
    )
    all_gpus = set(result["actor"]) | set(result["critic"]) | set(result["rollout"])
    assert all_gpus == {0, 1, 2, 3, 4, 5}
