"""
Solution for Exercise 02: Load Balancing Loss
"""

import torch


def load_balancing_loss(
    router_logits: torch.Tensor,
    routing_indices: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """
    Args:
        router_logits: (num_tokens, num_experts) - raw logits from the router
        routing_indices: (num_tokens, top_k) - indices of selected experts per token
        num_experts: total number of experts

    Returns:
        loss: scalar tensor - the load balancing loss
    """
    num_tokens = router_logits.shape[0]

    # f_i: fraction of tokens routed to each expert (hard assignment)
    # Count how many times each expert is selected across all tokens and top-k slots
    expert_counts = torch.zeros(num_experts, device=router_logits.device)
    for k in range(routing_indices.shape[1]):
        expert_counts.scatter_add_(
            0, routing_indices[:, k], torch.ones(num_tokens, device=router_logits.device)
        )
    total_assignments = routing_indices.shape[0] * routing_indices.shape[1]
    f = expert_counts / total_assignments  # (num_experts,)

    # P_i: mean router probability for each expert across all tokens
    router_probs = torch.softmax(router_logits, dim=-1)  # (num_tokens, num_experts)
    P = router_probs.mean(dim=0)  # (num_experts,)

    # Load balancing loss
    loss = num_experts * (f * P).sum()

    return loss
