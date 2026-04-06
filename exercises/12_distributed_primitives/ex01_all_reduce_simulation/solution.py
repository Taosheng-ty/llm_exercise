"""
Solution for Exercise 01: All-Reduce Simulation
"""

import torch


def simulate_all_reduce(
    worker_tensors: list[torch.Tensor], op: str
) -> list[torch.Tensor]:
    """
    Simulate all-reduce across virtual workers.
    """
    if op not in ("sum", "mean", "max"):
        raise ValueError(f"Unsupported op: {op}. Must be one of 'sum', 'mean', 'max'.")

    if len(worker_tensors) == 0:
        return []

    n = len(worker_tensors)
    stacked = torch.stack(worker_tensors)  # (N, *shape)

    if op == "sum":
        reduced = stacked.sum(dim=0)
    elif op == "mean":
        reduced = stacked.float().mean(dim=0).to(worker_tensors[0].dtype)
    elif op == "max":
        reduced = stacked.max(dim=0).values
    else:
        raise ValueError(f"Unknown op: {op}")

    # Every worker gets an identical clone of the reduced result
    return [reduced.clone() for _ in range(n)]
