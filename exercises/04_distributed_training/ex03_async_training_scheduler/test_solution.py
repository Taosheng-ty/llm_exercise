"""Tests for Exercise 03: Async Training Scheduler."""

import importlib.util
import os
import pytest

# Load solution from same directory
_spec = importlib.util.spec_from_file_location(
    "solution_ex03", os.path.join(os.path.dirname(__file__), "solution.py")
)
_sol = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sol)
Task = _sol.Task
schedule_tasks = _sol.schedule_tasks


def test_sequential_same_type():
    """Two generate tasks must run sequentially."""
    tasks = [
        Task("gen1", "generate", 5, []),
        Task("gen2", "generate", 3, []),
    ]
    schedule, makespan = schedule_tasks(tasks)
    assert makespan == 8
    # gen1 runs first (sorted by name for tie-breaking at time 0)
    assert schedule[0] == ("gen1", 0, 5)
    assert schedule[1] == ("gen2", 5, 8)


def test_overlap_different_types():
    """A generate and train task with no dependencies can overlap."""
    tasks = [
        Task("gen1", "generate", 5, []),
        Task("train1", "train", 3, []),
    ]
    schedule, makespan = schedule_tasks(tasks)
    # Both can start at time 0
    assert makespan == 5
    starts = {name: start for name, start, end in schedule}
    assert starts["gen1"] == 0
    assert starts["train1"] == 0


def test_async_prefetch_pattern():
    """Models the slime async pattern: gen(i+1) overlaps with train(i).

    gen0 -> train0 (depends on gen0)
    gen1 (no dep on train0, overlaps with train0)
    train1 (depends on gen1 and train0)
    """
    tasks = [
        Task("gen0", "generate", 4, []),
        Task("train0", "train", 6, ["gen0"]),
        Task("gen1", "generate", 4, ["gen0"]),
        Task("train1", "train", 6, ["gen1", "train0"]),
    ]
    schedule, makespan = schedule_tasks(tasks)
    times = {name: (start, end) for name, start, end in schedule}

    # gen0: 0-4
    assert times["gen0"] == (0, 4)
    # gen1 starts after gen0 (same resource): 4-8
    assert times["gen1"] == (4, 8)
    # train0 starts after gen0: 4-10
    assert times["train0"] == (4, 10)
    # train1 depends on gen1(done@8) and train0(done@10) and train resource(free@10)
    assert times["train1"] == (10, 16)
    assert makespan == 16


def test_dependency_chain():
    """Linear dependency chain should be fully sequential."""
    tasks = [
        Task("gen0", "generate", 3, []),
        Task("train0", "train", 4, ["gen0"]),
        Task("gen1", "generate", 3, ["train0"]),
        Task("train1", "train", 4, ["gen1"]),
    ]
    schedule, makespan = schedule_tasks(tasks)
    times = {name: (start, end) for name, start, end in schedule}
    assert times["gen0"] == (0, 3)
    assert times["train0"] == (3, 7)
    assert times["gen1"] == (7, 10)
    assert times["train1"] == (10, 14)
    assert makespan == 14


def test_empty_task_list():
    """Empty input should return empty schedule with zero makespan."""
    schedule, makespan = schedule_tasks([])
    assert schedule == []
    assert makespan == 0


def test_single_task():
    """Single task should start at time 0."""
    tasks = [Task("gen0", "generate", 10, [])]
    schedule, makespan = schedule_tasks(tasks)
    assert schedule == [("gen0", 0, 10)]
    assert makespan == 10


def test_three_stage_pipeline():
    """Three-stage async pipeline with overlap.

    gen0 -> train0
    gen1 (after gen0) -> train1 (after train0 and gen1)
    gen2 (after gen1) -> train2 (after train1 and gen2)
    """
    tasks = [
        Task("gen0", "generate", 2, []),
        Task("train0", "train", 3, ["gen0"]),
        Task("gen1", "generate", 2, ["gen0"]),
        Task("train1", "train", 3, ["gen1", "train0"]),
        Task("gen2", "generate", 2, ["gen1"]),
        Task("train2", "train", 3, ["gen2", "train1"]),
    ]
    schedule, makespan = schedule_tasks(tasks)
    times = {name: (start, end) for name, start, end in schedule}

    # gen0: 0-2, gen1: 2-4, gen2: 4-6
    assert times["gen0"] == (0, 2)
    assert times["gen1"] == (2, 4)
    assert times["gen2"] == (4, 6)
    # train0: 2-5, train1: max(4,5)=5 -> 5-8, train2: max(6,8)=8 -> 8-11
    assert times["train0"] == (2, 5)
    assert times["train1"] == (5, 8)
    assert times["train2"] == (8, 11)
    assert makespan == 11
