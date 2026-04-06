"""
Solution for Exercise 05: Shared + Routed Expert Architecture
"""

import torch
import torch.nn as nn


class ExpertFFN(nn.Module):
    """Simple 2-layer FFN expert: Linear -> ReLU -> Linear."""

    def __init__(self, dim: int, intermediate_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, intermediate_dim, bias=False)
        self.w2 = nn.Linear(intermediate_dim, dim, bias=False)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act(self.w1(x)))


class SharedMoELayer(nn.Module):
    """
    MoE layer with a shared expert + top-k routed experts.
    """

    def __init__(self, dim: int, num_routed_experts: int, top_k: int, intermediate_dim: int = None):
        super().__init__()
        if intermediate_dim is None:
            intermediate_dim = 4 * dim

        self.dim = dim
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k

        # Shared expert (always active)
        self.shared_expert = ExpertFFN(dim, intermediate_dim)

        # Routed experts
        self.routed_experts = nn.ModuleList(
            [ExpertFFN(dim, intermediate_dim) for _ in range(num_routed_experts)]
        )

        # Router: projects hidden states to num_routed_experts logits
        self.router = nn.Linear(dim, num_routed_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (num_tokens, dim)

        Returns:
            output: (num_tokens, dim)
        """
        # Shared expert output (always computed for all tokens)
        shared_output = self.shared_expert(hidden_states)

        # Routing
        router_logits = self.router(hidden_states)  # (num_tokens, num_routed_experts)
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = torch.softmax(top_k_logits, dim=-1)  # (num_tokens, top_k)

        # Compute routed expert outputs
        num_tokens = hidden_states.shape[0]
        routed_output = torch.zeros_like(hidden_states)

        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k]  # (num_tokens,)
            weights = routing_weights[:, k]  # (num_tokens,)

            for expert_id in range(self.num_routed_experts):
                mask = expert_indices == expert_id
                if not mask.any():
                    continue
                token_indices = mask.nonzero(as_tuple=True)[0]
                expert_input = hidden_states[token_indices]
                expert_out = self.routed_experts[expert_id](expert_input)
                routed_output[token_indices] += weights[token_indices].unsqueeze(1) * expert_out

        # Combine shared + routed
        output = shared_output + routed_output
        return output
