"""
Solution for Exercise 04: Cross-Attention
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """
    Multi-head cross-attention for encoder-decoder architectures.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Projection layers
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

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
        batch_size = decoder_hidden.size(0)
        decoder_len = decoder_hidden.size(1)
        encoder_len = encoder_output.size(1)

        # Project queries from decoder, keys/values from encoder
        Q = self.W_q(decoder_hidden)  # (B, T_dec, D)
        K = self.W_k(encoder_output)  # (B, T_enc, D)
        V = self.W_v(encoder_output)  # (B, T_enc, D)

        # Reshape to multi-head: (B, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, decoder_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, encoder_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, encoder_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores: (B, H, T_dec, T_enc)

        # Apply encoder mask: mask out padding positions in encoder
        if encoder_mask is not None:
            # encoder_mask: (B, T_enc) -> (B, 1, 1, T_enc) for broadcasting
            mask = encoder_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T_enc)
            scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)

        # Weighted sum of values
        context = torch.matmul(attn_weights, V)  # (B, H, T_dec, head_dim)

        # Reshape back: (B, T_dec, D)
        context = context.transpose(1, 2).contiguous().view(batch_size, decoder_len, self.d_model)

        # Output projection
        output = self.W_o(context)

        return output
