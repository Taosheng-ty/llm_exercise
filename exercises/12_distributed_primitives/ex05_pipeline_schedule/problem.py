"""
Exercise 05: Pipeline Parallel Schedule (Medium, numpy)

Simulate the GPipe pipeline parallel schedule. In pipeline parallelism, the
model is split into stages across GPUs. Micro-batches flow through stages
in a pipeline fashion.

GPipe schedule:
- All forward passes are completed first (stage 0 processes mb0 first,
  then when stage 0 starts mb1, stage 1 can start mb0, etc.).
- Then all backward passes run in reverse order: backward passes proceed
  from the last stage to the first, and within each stage the microbatches
  are processed in reverse order (last microbatch first). Specifically,
  the backward pass mirrors the forward pass -- the last forward event
  to finish is the first to start its backward pass.
- Each forward/backward step takes 1 time unit.

Implement the following functions:

    compute_gpipe_schedule(
        num_stages: int,
        num_microbatches: int,
    ) -> list[tuple[int, int, int, int, bool]]

    compute_bubble_ratio(
        num_stages: int,
        num_microbatches: int,
    ) -> float

compute_gpipe_schedule returns a list of events:
    (stage, microbatch, start_time, end_time, is_forward)

    sorted by (start_time, stage, not is_forward).

compute_bubble_ratio returns:
    bubble_ratio = idle_time / total_time

    where total_time = num_stages * total_wall_clock_time
    (i.e., total device-time across all stages)
    and idle_time = total_time - useful_work
    and useful_work = num_stages * num_microbatches * 2 (fwd + bwd each cost 1 unit)

Reference: GPipe paper, Megatron-LM pipeline scheduling.
"""

import numpy as np


def compute_gpipe_schedule(
    num_stages: int,
    num_microbatches: int,
) -> list[tuple[int, int, int, int, bool]]:
    """
    Compute GPipe pipeline schedule.

    Returns list of (stage, microbatch, start_time, end_time, is_forward).

    TODO: Implement this function.
    """
    raise NotImplementedError("Implement compute_gpipe_schedule")


def compute_bubble_ratio(
    num_stages: int,
    num_microbatches: int,
) -> float:
    """
    Compute the pipeline bubble ratio.

    TODO: Implement this function.
    """
    raise NotImplementedError("Implement compute_bubble_ratio")
