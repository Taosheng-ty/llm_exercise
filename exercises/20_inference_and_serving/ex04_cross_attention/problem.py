"""
Exercise 04: Cross-Attention (Medium)

Cross-attention is the mechanism that connects the encoder and decoder in
encoder-decoder architectures (e.g., T5, BART, Whisper). Unlike self-attention
where Q, K, V all come from the same sequence, in cross-attention:
  - Queries come from the decoder hidden states
  - Keys and Values come from the encoder output

Cross-attention enables encoder-decoder LLMs to condition generation on an encoded
input — essential for tasks like translation, summarization, and speech-to-text.
Understanding cross-attention is also important for multimodal LLMs where visual
encoders cross-attend with text decoders.

Key differences from self-attention:
- No causal mask on the encoder side (decoder can attend to all encoder positions)
- The encoder and decoder sequences can have different lengths
- Encoder tokens that are padding should be masked out

Your tasks:
-----------
1. Implement `CrossAttention(d_model, num_heads)` as an nn.Module:
   - Linear projections: W_q (decoder -> queries), W_k (encoder -> keys),
     W_v (encoder -> values), W_o (output projection)
   - `forward(decoder_hidden, encoder_output, encoder_mask=None) -> output`:
     - decoder_hidden: (batch, decoder_len, d_model)
     - encoder_output: (batch, encoder_len, d_model)
     - encoder_mask: (batch, encoder_len) boolean, True = valid token, False = pad
     - Returns: (batch, decoder_len, d_model)
   - Use multi-head attention with scaled dot-product
   - Apply mask before softmax: set padded positions to -inf
"""

import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    """
    Multi-head cross-attention for encoder-decoder architectures.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        raise NotImplementedError("Implement CrossAttention.__init__")

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            decoder_hidden: (batch, decoder_len, d_model) - queries source
            encoder_output: (batch, encoder_len, d_model) - keys/values source
            encoder_mask: (batch, encoder_len) - True for valid, False for padding

        Returns:
            Output of shape (batch, decoder_len, d_model)
        """
        raise NotImplementedError("Implement CrossAttention.forward")
