"""
Solution for Exercise 01: Top-K Expert Routing
"""

import torch


def top_k_routing(
    hidden_states: torch.Tensor,
    router_weights: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        hidden_states: (num_tokens, dim) - token representations
        router_weights: (dim, num_experts) - router projection matrix
        top_k: number of experts to select per token

    Returns:
        routing_weights: (num_tokens, top_k) - softmax weights for selected experts
        routing_indices: (num_tokens, top_k) - indices of selected experts
    """
    # Step 1: Compute router logits
    router_logits = hidden_states @ router_weights  # (num_tokens, num_experts)

    # Step 2: Select top-k experts per token
    top_k_logits, routing_indices = torch.topk(router_logits, top_k, dim=-1)

    # Step 3: Apply softmax only over selected top-k logits
    routing_weights = torch.softmax(top_k_logits, dim=-1)

    return routing_weights, routing_indices
