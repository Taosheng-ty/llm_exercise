"""
Solution for Exercise 04: Routing Decision Caching (Routing Replay)

Based on slime's routing_replay.py RoutingReplay class.
"""

import torch


class RoutingReplayCache:
    def __init__(self):
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list = []

    def record(self, top_indices: torch.Tensor) -> None:
        """Record routing decision. Store a copy (not a reference)."""
        self.top_indices_list.append(top_indices.clone().detach())

    def replay_forward(self) -> torch.Tensor:
        """Return the next recorded routing decision for forward replay."""
        top_indices = self.top_indices_list[self.forward_index]
        self.forward_index += 1
        return top_indices

    def replay_backward(self) -> torch.Tensor:
        """Return the next recorded routing decision for backward replay."""
        top_indices = self.top_indices_list[self.backward_index]
        self.backward_index += 1
        return top_indices

    def clear(self) -> None:
        """Reset all indices and clear stored decisions."""
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list = []

    def clear_forward(self) -> None:
        """Reset only the forward replay index."""
        self.forward_index = 0


def compute_topk_with_replay(
    scores: torch.Tensor,
    topk: int,
    cache: RoutingReplayCache,
    stage: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        scores: (num_tokens, num_experts) - router scores
        topk: number of experts to select
        cache: RoutingReplayCache instance
        stage: one of "record", "replay_forward", "replay_backward", "fallthrough"

    Returns:
        probs: (num_tokens, topk) - routing probabilities for selected experts
        top_indices: (num_tokens, topk) - selected expert indices
    """
    if stage == "fallthrough":
        probs, top_indices = torch.topk(scores, topk, dim=-1)
        return probs, top_indices

    elif stage == "record":
        probs, top_indices = torch.topk(scores, topk, dim=-1)
        cache.record(top_indices)
        return probs, top_indices

    elif stage == "replay_forward":
        top_indices = cache.replay_forward()
        probs = scores.gather(1, top_indices)
        return probs, top_indices

    elif stage == "replay_backward":
        top_indices = cache.replay_backward()
        probs = scores.gather(1, top_indices)
        return probs, top_indices

    else:
        raise ValueError(f"Unknown stage: {stage}")
