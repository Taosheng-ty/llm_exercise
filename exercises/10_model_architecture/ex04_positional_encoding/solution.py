"""
Solution for Exercise 04: Sinusoidal Positional Encoding
"""

import torch
import math


def sinusoidal_positional_encoding(max_seq_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_seq_len, d_model)
    position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)  # (max_seq_len, 1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
    )  # (d_model // 2,)

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
