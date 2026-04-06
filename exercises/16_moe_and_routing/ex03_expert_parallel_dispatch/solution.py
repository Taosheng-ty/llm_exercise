"""
Solution for Exercise 03: Expert Parallel Token Dispatch
"""

import torch


def expert_dispatch(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    routing_indices: torch.Tensor,
    expert_ffns: list,
    num_experts: int,
) -> torch.Tensor:
    """
    Args:
        hidden_states: (num_tokens, dim) - input token representations
        routing_weights: (num_tokens, top_k) - softmax weights for selected experts
        routing_indices: (num_tokens, top_k) - expert indices per token
        expert_ffns: list of num_experts callables, each (batch, dim) -> (batch, dim)
        num_experts: total number of experts

    Returns:
        output: (num_tokens, dim) - combined expert outputs
    """
    num_tokens, dim = hidden_states.shape
    top_k = routing_indices.shape[1]

    output = torch.zeros_like(hidden_states)

    for expert_id in range(num_experts):
        # Find all (token, slot) pairs assigned to this expert
        mask = routing_indices == expert_id  # (num_tokens, top_k)

        if not mask.any():
            continue  # No tokens for this expert

        # For each top-k slot, process tokens assigned to this expert
        for k in range(top_k):
            slot_mask = mask[:, k]  # (num_tokens,) bool
            if not slot_mask.any():
                continue

            # Gather tokens for this expert
            token_indices = slot_mask.nonzero(as_tuple=True)[0]
            expert_input = hidden_states[token_indices]  # (batch_expert, dim)

            # Process through expert FFN
            expert_output = expert_ffns[expert_id](expert_input)  # (batch_expert, dim)

            # Weighted accumulation
            weights = routing_weights[token_indices, k].unsqueeze(1)  # (batch_expert, 1)
            output[token_indices] += weights * expert_output

    return output
