"""
Exercise 07: Expert Capacity Factor Limiting (Medium, numpy)

In MoE models, some experts may receive far more tokens than others, causing
memory and compute imbalance. The capacity factor mechanism limits the maximum
number of tokens each expert can process:

    max_tokens_per_expert = capacity_factor * (total_assignments / num_experts)

where total_assignments = num_tokens * top_k (each token creates top_k
assignments). When top_k=1, total_assignments == num_tokens.

The capacity factor prevents any single expert from being overwhelmed during MoE
LLM training. Without capacity limits, popular experts receive far more tokens
than they can process efficiently, causing memory spikes and load imbalance
across GPUs. Dropped tokens are a controlled tradeoff for stable, efficient
distributed training.

Tokens that exceed an expert's capacity are "dropped" (their output is zero).

Your task:
    Implement `apply_capacity_factor(routing_decisions, routing_weights,
                                      num_experts, capacity_factor)` that:
    1. Computes the capacity limit per expert
    2. For each expert, keeps only the first `capacity` tokens (in order)
    3. Drops (zeros out weights of) excess tokens
    4. Returns:
       - adjusted_weights: same shape as routing_weights, with dropped tokens zeroed
       - drop_rate: float, fraction of total assignments that were dropped
"""

import numpy as np


def apply_capacity_factor(
    routing_decisions: np.ndarray,
    routing_weights: np.ndarray,
    num_experts: int,
    capacity_factor: float,
) -> tuple[np.ndarray, float]:
    """
    Args:
        routing_decisions: (num_tokens, top_k) int - selected expert indices
        routing_weights: (num_tokens, top_k) float - routing weights
        num_experts: total number of experts
        capacity_factor: multiplier for capacity (1.0 = exactly balanced capacity)

    Returns:
        adjusted_weights: (num_tokens, top_k) float - weights with dropped tokens zeroed
        drop_rate: float - fraction of assignments that were dropped
    """
    raise NotImplementedError("Implement capacity factor limiting")
