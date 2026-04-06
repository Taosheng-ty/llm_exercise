"""
Solution for Exercise 07: Reorder MoE Expert Weights
"""

import numpy as np


def compute_expert_order(
    routing_scores: np.ndarray,
) -> np.ndarray:
    """Compute new expert ordering based on routing frequency (descending)."""
    # Sum routing scores across all tokens for each expert
    total_scores = routing_scores.sum(axis=0)  # shape: (num_experts,)
    # Sort by descending score
    new_order = np.argsort(-total_scores)
    return new_order


def reorder_expert_weights(
    expert_weights: dict[int, np.ndarray],
    new_order: np.ndarray,
) -> dict[int, np.ndarray]:
    """Reorder expert weights according to new_order."""
    reordered = {}
    for new_idx, old_idx in enumerate(new_order):
        reordered[new_idx] = expert_weights[int(old_idx)].copy()
    return reordered


def reorder_router_weights(
    router_weight: np.ndarray,
    new_order: np.ndarray,
) -> np.ndarray:
    """Reorder router weight rows to match new expert order."""
    return router_weight[new_order].copy()


def reorder_moe_layer(
    expert_weights: dict[int, np.ndarray],
    router_weight: np.ndarray,
    routing_scores: np.ndarray,
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """Reorder an entire MoE layer by expert frequency."""
    new_order = compute_expert_order(routing_scores)
    reordered_experts = reorder_expert_weights(expert_weights, new_order)
    reordered_router = reorder_router_weights(router_weight, new_order)
    return reordered_experts, reordered_router, new_order
