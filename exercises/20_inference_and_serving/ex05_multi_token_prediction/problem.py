"""
Exercise 05: Multi-Token Prediction (Medium)

Standard language models predict one token at a time: given hidden state h_t,
predict token t+1. Multi-token prediction (MTP), as explored by Meta and
DeepSeek, trains the model to predict multiple future tokens simultaneously
from the same hidden state.

Each "prediction head" is a separate linear projection from hidden states to
vocabulary logits. Head i predicts the token at position t+i (i=1..N).

Benefits:
- Richer training signal per forward pass
- Can improve sample efficiency
- Enables speculative decoding at inference time

Your tasks:
-----------
1. Implement `MultiTokenPredictionHead(d_model, vocab_size, num_futures)` as nn.Module:
   - Store the heads in `self.heads` as an nn.ModuleList of nn.Linear layers.
   - Each head maps from d_model -> vocab_size.
   - `forward(hidden_states) -> list[Tensor]`:
     - hidden_states: (batch, seq_len, d_model)
     - Returns list of `num_futures` tensors, each (batch, seq_len, vocab_size)
     - predictions[i] represents logits for position t+i+1

2. Implement `compute_multi_token_loss(predictions, targets, mask) -> (total_loss, per_head_losses)`:
   - predictions: list of N tensors, each (batch, seq_len, vocab_size)
   - targets: (batch, seq_len) of token IDs -- the full target sequence
   - mask: (batch, seq_len) boolean, True = compute loss here
   - For head i (0-indexed), the target at position t is targets[t+i+1]
     (i.e., head 0 predicts next token, head 1 predicts token after that, etc.)
   - Only compute loss where mask is True AND the shifted target position exists.
   - Return (total_loss: scalar mean, per_head_losses: list of N scalars)
"""

import torch
import torch.nn as nn


class MultiTokenPredictionHead(nn.Module):
    """N-ahead prediction heads from a single hidden state."""

    def __init__(self, d_model: int, vocab_size: int, num_futures: int):
        super().__init__()
        raise NotImplementedError("Implement MultiTokenPredictionHead.__init__")

    def forward(self, hidden_states: torch.Tensor) -> list:
        """
        Args:
            hidden_states: (batch, seq_len, d_model)

        Returns:
            List of num_futures tensors, each (batch, seq_len, vocab_size)
        """
        raise NotImplementedError("Implement MultiTokenPredictionHead.forward")


def compute_multi_token_loss(
    predictions: list,
    targets: torch.Tensor,
    mask: torch.Tensor,
):
    """
    Compute cross-entropy loss for each prediction head.

    Args:
        predictions: list of N tensors, each (batch, seq_len, vocab_size)
        targets: (batch, seq_len) token IDs
        mask: (batch, seq_len) boolean mask

    Returns:
        (total_loss, per_head_losses) where total_loss is scalar mean,
        per_head_losses is list of N scalar losses.
    """
    raise NotImplementedError("Implement compute_multi_token_loss")
