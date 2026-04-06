"""
Exercise 01: All-Reduce Simulation (Medium, PyTorch)

Simulate the all-reduce collective operation across N virtual workers
using only single-process PyTorch (no actual distributed runtime needed).

In real distributed training, all-reduce aggregates tensors across all workers
so that every worker ends up with the same reduced result. Here we simulate
this by representing each worker's tensor as an element in a Python list.

Reference: slime distributed_utils.py patterns for all_reduce usage.

Implement the following function:

    simulate_all_reduce(worker_tensors: list[torch.Tensor], op: str) -> list[torch.Tensor]

Args:
    worker_tensors: A list of N tensors (one per virtual worker). All tensors
                    have the same shape and dtype.
    op: One of "sum", "mean", "max".

Returns:
    A list of N tensors where each tensor is the result of the reduction.
    After all-reduce, every worker holds an identical copy of the reduced tensor.

Example:
    worker_tensors = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    result = simulate_all_reduce(worker_tensors, "sum")
    # result == [torch.tensor([4.0, 6.0]), torch.tensor([4.0, 6.0])]
"""

import torch


def simulate_all_reduce(
    worker_tensors: list[torch.Tensor], op: str
) -> list[torch.Tensor]:
    """
    Simulate all-reduce across virtual workers.

    TODO: Implement this function.
    1. Validate that op is one of "sum", "mean", "max".
    2. Compute the reduced tensor according to op.
    3. Return a list where every worker gets a clone of the reduced result.
    """
    raise NotImplementedError("Implement simulate_all_reduce")
