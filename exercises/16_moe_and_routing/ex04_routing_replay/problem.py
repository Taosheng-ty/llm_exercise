"""
Exercise 04: Routing Decision Caching (Routing Replay) (Medium, PyTorch)

In reinforcement learning from human feedback (RLHF) training of MoE models,
we often need to replay the same routing decisions during the backward pass
that were made during the forward pass. This is critical for off-policy training
where the model parameters change between the forward and backward passes.

Reference: slime's routing_replay.py RoutingReplay class, which records routing
decisions during forward pass and replays them during backward pass, offloading
to CPU pinned memory for efficiency.

Your task:
    Implement the RoutingReplayCache class with:
    - record(top_indices): Store routing decisions (top-k expert indices per token)
    - replay_forward(): Return the next recorded routing decision for forward replay
    - replay_backward(): Return the next recorded routing decision for backward replay
    - clear(): Reset all state
    - clear_forward(): Reset only the forward replay index

    Also implement compute_topk_with_replay(scores, topk, cache, stage) that:
    - "record": Computes real top-k, records to cache, returns (probs, indices)
    - "replay_forward": Replays indices from cache, gathers probs from scores
    - "replay_backward": Same as replay_forward but uses backward index
    - "fallthrough": Just computes normal top-k without caching
"""

import torch


class RoutingReplayCache:
    def __init__(self):
        raise NotImplementedError("Implement RoutingReplayCache")

    def record(self, top_indices: torch.Tensor) -> None:
        """Record routing decision. Store a copy (not a reference)."""
        raise NotImplementedError

    def replay_forward(self) -> torch.Tensor:
        """Return the next recorded routing decision for forward replay."""
        raise NotImplementedError

    def replay_backward(self) -> torch.Tensor:
        """Return the next recorded routing decision for backward replay."""
        raise NotImplementedError

    def clear(self) -> None:
        """Reset all indices and clear stored decisions."""
        raise NotImplementedError

    def clear_forward(self) -> None:
        """Reset only the forward replay index."""
        raise NotImplementedError


def compute_topk_with_replay(
    scores: torch.Tensor,
    topk: int,
    cache: RoutingReplayCache,
    stage: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        scores: (num_tokens, num_experts) - router scores (after softmax or raw)
        topk: number of experts to select
        cache: RoutingReplayCache instance
        stage: one of "record", "replay_forward", "replay_backward", "fallthrough"

    Returns:
        probs: (num_tokens, topk) - routing probabilities for selected experts
        top_indices: (num_tokens, topk) - selected expert indices
    """
    raise NotImplementedError("Implement compute_topk_with_replay")
