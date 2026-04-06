"""
Solution for Exercise 06: Expert Frequency Analysis
"""

import numpy as np


def expert_utilization_rate(
    routing_decisions: np.ndarray,
    num_experts: int,
) -> np.ndarray:
    """
    Args:
        routing_decisions: (num_tokens, top_k) - expert indices selected per token
        num_experts: total number of experts

    Returns:
        utilization: (num_experts,) - fraction of tokens routed to each expert
    """
    num_tokens = routing_decisions.shape[0]
    utilization = np.zeros(num_experts)

    for expert_id in range(num_experts):
        # Count tokens that have this expert in any of their top-k slots
        tokens_using_expert = np.any(routing_decisions == expert_id, axis=1).sum()
        utilization[expert_id] = tokens_using_expert / num_tokens

    return utilization


def load_imbalance_score(
    routing_decisions: np.ndarray,
    num_experts: int,
) -> float:
    """
    Args:
        routing_decisions: (num_tokens, top_k)
        num_experts: total number of experts

    Returns:
        imbalance: float - coefficient of variation (std/mean) of utilization
    """
    utilization = expert_utilization_rate(routing_decisions, num_experts)
    mean_util = utilization.mean()

    if mean_util == 0:
        return float("inf")

    return float(utilization.std() / mean_util)


def find_dead_experts(
    routing_decisions: np.ndarray,
    num_experts: int,
    threshold: float = 0.01,
) -> list[int]:
    """
    Args:
        routing_decisions: (num_tokens, top_k)
        num_experts: total number of experts
        threshold: experts with utilization below this are considered dead

    Returns:
        dead_experts: sorted list of expert indices with utilization < threshold
    """
    utilization = expert_utilization_rate(routing_decisions, num_experts)
    dead = [int(i) for i in range(num_experts) if utilization[i] < threshold]
    return sorted(dead)
