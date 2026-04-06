"""
Exercise 04: Sinusoidal Positional Encoding
=============================================
Difficulty: Easy

The original Transformer (Vaswani et al., 2017) uses sinusoidal positional encodings
to inject position information into token embeddings. Positional encoding gives
transformer LLMs the ability to understand token order — without it, "the cat sat on
the mat" and "the mat sat on the cat" would be indistinguishable. While modern LLMs
have moved to RoPE, understanding sinusoidal encoding is foundational for grasping how
position information flows through the model.

Formula:
    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

Your task:
    Implement sinusoidal_positional_encoding() that returns a (max_seq_len, d_model) tensor.
"""

import torch


def sinusoidal_positional_encoding(max_seq_len: int, d_model: int) -> torch.Tensor:
    """
    Compute sinusoidal positional encoding.

    Args:
        max_seq_len: maximum sequence length
        d_model: model dimension (must be even)

    Returns:
        Tensor of shape (max_seq_len, d_model) with positional encodings
    """
    # TODO:
    # 1. Create a position vector: [0, 1, ..., max_seq_len-1] shaped (max_seq_len, 1)
    # 2. Create a dimension vector for the even indices: [0, 2, 4, ...] shaped (1, d_model//2)
    # 3. Compute the angle rates: pos / 10000^(dim / d_model)
    # 4. Fill even columns with sin, odd columns with cos
    raise NotImplementedError("Implement sinusoidal_positional_encoding")
