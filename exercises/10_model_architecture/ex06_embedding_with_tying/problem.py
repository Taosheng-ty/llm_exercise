"""
Exercise 06: Tied Input/Output Embeddings
==========================================
Difficulty: Medium

Many language models tie (share) the input embedding matrix with the output projection
(LM head). This means the same weight matrix is used for:

    1. Looking up token embeddings: embed = weight[input_ids]
    2. Projecting hidden states to logits: logits = hidden @ weight^T

This reduces parameters and can improve training.

Your task:
    Implement TiedEmbedding that:
    1. Has a single weight matrix of shape (vocab_size, embed_dim)
    2. embed(input_ids) -> lookup embeddings
    3. project(hidden_states) -> compute logits via matmul with same weight
"""

import torch
import torch.nn as nn


class TiedEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        """
        Args:
            vocab_size: vocabulary size
            embed_dim: embedding dimension
        """
        super().__init__()
        # TODO: create a single weight parameter of shape (vocab_size, embed_dim)
        # Initialize from normal distribution with std=0.02
        raise NotImplementedError("Implement __init__")

    def embed(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Look up embeddings for input_ids.

        Args:
            input_ids: (batch, seq_len) of token IDs
        Returns:
            (batch, seq_len, embed_dim) embeddings
        """
        # TODO: use the weight as a lookup table
        raise NotImplementedError("Implement embed")

    def project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Project hidden states to logits using the SAME weight matrix.

        Args:
            hidden_states: (batch, seq_len, embed_dim)
        Returns:
            (batch, seq_len, vocab_size) logits
        """
        # TODO: matmul hidden_states with weight^T
        raise NotImplementedError("Implement project")
