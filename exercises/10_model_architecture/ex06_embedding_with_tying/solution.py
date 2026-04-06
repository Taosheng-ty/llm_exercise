"""
Solution for Exercise 06: Tied Input/Output Embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TiedEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, embed_dim) * 0.02)

    def embed(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return F.embedding(input_ids, self.weight)

    def project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states @ self.weight.T
