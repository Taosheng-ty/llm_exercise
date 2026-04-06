"""
Exercise 08: Minimal Language Model
=====================================
Difficulty: Medium

Build a minimal GPT-style language model with:
    Embedding -> N x TransformerBlock -> RMSNorm -> Linear (LM Head)

The model should support:
    1. forward(input_ids) -> logits
    2. generate(input_ids, max_new_tokens) -> generated token ids (greedy decoding)

Your task:
    Implement SimpleLM using components from previous exercises (or reimplement them).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# You may copy/reimplement these from previous exercises
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, ffn_hidden_dim):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class SimpleLM(nn.Module):
    def __init__(self, vocab_size: int, dim: int, n_layers: int, n_heads: int,
                 ffn_hidden_dim: int, max_seq_len: int = 512):
        """
        Args:
            vocab_size: size of vocabulary
            dim: model dimension
            n_layers: number of transformer blocks
            n_heads: number of attention heads
            ffn_hidden_dim: FFN hidden dimension
            max_seq_len: maximum sequence length
        """
        super().__init__()
        # TODO:
        # 1. Token embedding: nn.Embedding(vocab_size, dim)
        # 2. Stack of n_layers TransformerBlocks
        # 3. Final RMSNorm
        # 4. LM head: nn.Linear(dim, vocab_size, bias=False)
        raise NotImplementedError("Implement __init__")

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len)
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        # TODO:
        # 1. Embed tokens
        # 2. Pass through transformer blocks
        # 3. Apply final norm
        # 4. Project to vocab
        raise NotImplementedError("Implement forward")

    @torch.no_grad()
    def generate(self, input_ids: torch.LongTensor, max_new_tokens: int) -> torch.LongTensor:
        """
        Greedy autoregressive generation.

        Args:
            input_ids: (batch, seq_len) prompt tokens
            max_new_tokens: number of tokens to generate
        Returns:
            (batch, seq_len + max_new_tokens) tensor with generated tokens appended
        """
        # TODO:
        # For each new token:
        #   1. Forward pass on current sequence
        #   2. Take logits at last position
        #   3. Argmax to get next token
        #   4. Append to sequence
        raise NotImplementedError("Implement generate")
