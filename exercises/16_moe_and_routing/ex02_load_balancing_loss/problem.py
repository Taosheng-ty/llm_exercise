"""
Exercise 02: Load Balancing Loss (Medium, PyTorch)

Implement the auxiliary load balancing loss used in MoE models to encourage
uniform expert utilization. Without this loss, MoE models tend to collapse
to using only a few experts.

The load balancing loss is defined as:
    loss = num_experts * sum_i(f_i * P_i)

where:
    f_i = fraction of tokens routed to expert i (based on hard routing decisions)
    P_i = mean router probability assigned to expert i (based on soft logits)

This loss is minimized when routing is perfectly uniform across experts.

Your task:
    Implement `load_balancing_loss(router_logits, routing_indices, num_experts)`
"""

import torch


def load_balancing_loss(
    router_logits: torch.Tensor,
    routing_indices: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """
    Compute the auxiliary load balancing loss for MoE routing.

    Args:
        router_logits: (num_tokens, num_experts) - raw logits from the router
        routing_indices: (num_tokens, top_k) - indices of selected experts per token
        num_experts: total number of experts

    Returns:
        loss: scalar tensor - the load balancing loss

    How to compute f_i and P_i:
        f_i (expert load fraction): Count how many times expert i appears in
            routing_indices (across all tokens and all top_k slots), then divide
            by total number of assignments (num_tokens * top_k).
            Hint: use one_hot on routing_indices, sum over tokens and top_k dim.

        P_i (mean router probability for expert i): Apply softmax to router_logits
            to get per-token probabilities, then take the mean across all tokens
            for each expert.

        loss = num_experts * sum(f * P)  (dot product of f and P vectors)
    """
    raise NotImplementedError("Implement load balancing loss")
