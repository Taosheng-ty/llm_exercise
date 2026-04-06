"""
Solution for Exercise 05: Pipeline Parallel Schedule
"""

import numpy as np


def compute_gpipe_schedule(
    num_stages: int,
    num_microbatches: int,
) -> list[tuple[int, int, int, int, bool]]:
    """
    Compute GPipe pipeline schedule.

    GPipe: all forwards first, then all backwards.

    Forward pass for (stage s, microbatch m) starts at time s + m
    (each microbatch must wait for the previous stage to finish it,
    and for the same stage to finish the previous microbatch).

    After all forwards complete, backward passes run in reverse stage order.
    Backward for (stage s, microbatch m) in GPipe:
    - The last forward ends at time (num_stages - 1) + (num_microbatches - 1) + 1
      = num_stages + num_microbatches - 2 + 1 = num_stages + num_microbatches - 1.
      That is fwd_end.
    - Backwards go from last stage to first. Within a stage, microbatches are
      processed in order (0, 1, 2, ...).
    - backward(stage s, mb m) starts at:
      fwd_end + (num_stages - 1 - s) + m
    """
    events = []
    fwd_end = num_stages + num_microbatches - 1  # time when all fwd done

    # Forward passes
    for m in range(num_microbatches):
        for s in range(num_stages):
            start = s + m
            end = start + 1
            events.append((s, m, start, end, True))

    # Backward passes: reverse stage order, microbatches in order
    for m in range(num_microbatches):
        for s in range(num_stages - 1, -1, -1):
            start = fwd_end + (num_stages - 1 - s) + m
            end = start + 1
            events.append((s, m, start, end, False))

    # Sort by (start_time, stage, backward before forward at same time)
    events.sort(key=lambda e: (e[2], e[0], not e[4]))
    return events


def compute_bubble_ratio(
    num_stages: int,
    num_microbatches: int,
) -> float:
    """
    Compute the pipeline bubble ratio.

    Total wall clock time: from t=0 to the end of the last backward.
    Last backward ends at: fwd_end + (num_stages - 1) + (num_microbatches - 1) + 1
        = (num_stages + num_microbatches - 1) + num_stages - 1 + num_microbatches - 1 + 1
        = 2 * num_stages + 2 * num_microbatches - 2

    Useful work per stage: num_microbatches * 2 (fwd + bwd, each 1 time unit)
    Total useful work: num_stages * num_microbatches * 2
    Total device time: num_stages * wall_clock_time
    Bubble = total_device_time - useful_work
    Bubble ratio = bubble / total_device_time
    """
    wall_clock = 2 * (num_stages + num_microbatches - 1)
    total_device_time = num_stages * wall_clock
    useful_work = num_stages * num_microbatches * 2
    idle_time = total_device_time - useful_work
    return idle_time / total_device_time
