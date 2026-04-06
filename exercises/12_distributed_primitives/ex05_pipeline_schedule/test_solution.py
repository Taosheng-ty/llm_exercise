"""
Tests for Exercise 05: Pipeline Parallel Schedule
"""

import importlib.util
import os

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

compute_gpipe_schedule = _mod.compute_gpipe_schedule
compute_bubble_ratio = _mod.compute_bubble_ratio


def test_schedule_event_count():
    """Total events = num_stages * num_microbatches * 2 (fwd + bwd)."""
    events = compute_gpipe_schedule(4, 8)
    assert len(events) == 4 * 8 * 2


def test_schedule_simple_2x2():
    events = compute_gpipe_schedule(2, 2)
    # Expected:
    # fwd(s0,m0) t=0-1, fwd(s0,m1) t=1-2, fwd(s1,m0) t=1-2, fwd(s1,m1) t=2-3
    # bwd(s1,m0) t=3-4, bwd(s1,m1) t=4-5, bwd(s0,m0) t=4-5, bwd(s0,m1) t=5-6
    fwd_events = [(s, m, st, en) for s, m, st, en, is_fwd in events if is_fwd]
    bwd_events = [(s, m, st, en) for s, m, st, en, is_fwd in events if not is_fwd]

    assert len(fwd_events) == 4
    assert len(bwd_events) == 4

    # Check forward: stage 0 mb 0 starts at t=0
    assert (0, 0, 0, 1) in fwd_events
    # Stage 1 mb 0 starts at t=1
    assert (1, 0, 1, 2) in fwd_events
    # Stage 1 mb 1 starts at t=2
    assert (1, 1, 2, 3) in fwd_events

    # All forwards complete at t=3, backwards start at t=3
    # bwd(s1, m0) at t=3-4
    assert (1, 0, 3, 4) in bwd_events
    # bwd(s0, m0) at t=4-5
    assert (0, 0, 4, 5) in bwd_events


def test_no_stage_overlap():
    """A stage should not have overlapping events."""
    events = compute_gpipe_schedule(4, 6)
    from collections import defaultdict

    stage_events = defaultdict(list)
    for s, m, st, en, is_fwd in events:
        stage_events[s].append((st, en))

    for s, intervals in stage_events.items():
        intervals.sort()
        for i in range(len(intervals) - 1):
            assert intervals[i][1] <= intervals[i + 1][0], (
                f"Stage {s} has overlapping events: {intervals[i]} and {intervals[i+1]}"
            )


def test_forward_before_backward():
    """For each (stage, microbatch), forward must complete before backward starts."""
    events = compute_gpipe_schedule(3, 4)
    fwd_map = {}
    bwd_map = {}
    for s, m, st, en, is_fwd in events:
        if is_fwd:
            fwd_map[(s, m)] = (st, en)
        else:
            bwd_map[(s, m)] = (st, en)

    for key in fwd_map:
        assert fwd_map[key][1] <= bwd_map[key][0], (
            f"Forward for {key} ends at {fwd_map[key][1]} but backward starts at {bwd_map[key][0]}"
        )


def test_dependency_forward():
    """Forward(s, m) must start after forward(s-1, m) finishes."""
    events = compute_gpipe_schedule(4, 4)
    fwd_map = {}
    for s, m, st, en, is_fwd in events:
        if is_fwd:
            fwd_map[(s, m)] = (st, en)

    for s in range(1, 4):
        for m in range(4):
            assert fwd_map[(s - 1, m)][1] <= fwd_map[(s, m)][0]


def test_bubble_ratio_formula():
    """Verify bubble ratio for known cases."""
    # With 4 stages, 4 microbatches:
    # wall_clock = 2*(4+4-1) = 14
    # total_device_time = 4 * 14 = 56
    # useful_work = 4 * 4 * 2 = 32
    # bubble = 24, ratio = 24/56 = 3/7
    ratio = compute_bubble_ratio(4, 4)
    assert abs(ratio - 24 / 56) < 1e-10


def test_bubble_ratio_decreases_with_microbatches():
    """More microbatches should reduce bubble ratio."""
    r1 = compute_bubble_ratio(4, 4)
    r2 = compute_bubble_ratio(4, 16)
    r3 = compute_bubble_ratio(4, 64)
    assert r1 > r2 > r3


def test_bubble_ratio_single_stage():
    """With 1 stage, there is no bubble."""
    ratio = compute_bubble_ratio(1, 10)
    assert abs(ratio) < 1e-10


def test_all_events_duration_one():
    """Each event should have duration of exactly 1 time unit."""
    events = compute_gpipe_schedule(3, 5)
    for s, m, st, en, is_fwd in events:
        assert en - st == 1
