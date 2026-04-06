"""
Solution for Exercise 07: Expert Capacity Factor Limiting
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
    num_tokens, top_k = routing_decisions.shape
    total_assignments = num_tokens * top_k

    # Compute capacity per expert
    capacity = int(capacity_factor * (num_tokens * top_k / num_experts))
    capacity = max(capacity, 1)  # At least 1 token per expert

    adjusted_weights = routing_weights.copy()
    total_dropped = 0

    # Track how many tokens each expert has received so far
    expert_counts = np.zeros(num_experts, dtype=int)

    # Process tokens in order (token 0 slot 0, token 0 slot 1, ..., token 1 slot 0, ...)
    for token_idx in range(num_tokens):
        for k in range(top_k):
            expert_id = routing_decisions[token_idx, k]
            if expert_counts[expert_id] < capacity:
                expert_counts[expert_id] += 1
            else:
                # Drop this assignment
                adjusted_weights[token_idx, k] = 0.0
                total_dropped += 1

    drop_rate = total_dropped / total_assignments if total_assignments > 0 else 0.0

    return adjusted_weights, drop_rate
