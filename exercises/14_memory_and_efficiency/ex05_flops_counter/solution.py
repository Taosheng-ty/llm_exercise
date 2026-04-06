"""
Solution for Exercise 05: FLOPs Counter
"""


def linear_flops(
    batch_size: int, seq_len: int, in_features: int, out_features: int
) -> int:
    """FLOPs for a linear layer: 2 * B * S * in * out."""
    return 2 * batch_size * seq_len * in_features * out_features


def attention_flops(
    batch_size: int, num_heads: int, seq_len: int, head_dim: int
) -> int:
    """
    FLOPs for multi-head attention (causal).
    QK^T (causal): 2 * H * S * S * D / 2 = H * S * S * D
    softmax*V: H * S * S * D
    Total per sample: 2 * H * S * S * D
    """
    # QK^T with causal mask (half the entries)
    qk_flops = 2 * num_heads * seq_len * seq_len * head_dim // 2
    # Attention weights * V
    av_flops = num_heads * seq_len * seq_len * head_dim
    return batch_size * (qk_flops + av_flops)


def ffn_flops(
    batch_size: int, seq_len: int, hidden_dim: int, ffn_hidden_dim: int
) -> int:
    """
    FLOPs for SwiGLU FFN: 3 projections (gate, up, down).
    Each is a linear: 2 * B * S * hidden * ffn_hidden
    Total: 2 * B * S * hidden * ffn_hidden * 3
    """
    return 2 * batch_size * seq_len * hidden_dim * ffn_hidden_dim * 3


def transformer_block_flops(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_heads: int,
    ffn_hidden_dim: int,
) -> int:
    """FLOPs for one transformer block."""
    head_dim = hidden_dim // num_heads

    # QKV projections: 3 linear layers from hidden_dim to hidden_dim
    qkv = 3 * linear_flops(batch_size, seq_len, hidden_dim, hidden_dim)

    # Attention computation
    attn = attention_flops(batch_size, num_heads, seq_len, head_dim)

    # Output projection
    out_proj = linear_flops(batch_size, seq_len, hidden_dim, hidden_dim)

    # FFN
    ffn = ffn_flops(batch_size, seq_len, hidden_dim, ffn_hidden_dim)

    return qkv + attn + out_proj + ffn


def total_training_flops(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_heads: int,
    ffn_hidden_dim: int,
    num_layers: int,
    vocab_size: int,
) -> int:
    """
    Total training FLOPs = 3 * forward FLOPs.
    Forward FLOPs = num_layers * block_flops + lm_head_flops
    """
    block = transformer_block_flops(
        batch_size, seq_len, hidden_dim, num_heads, ffn_hidden_dim
    )

    # LM head: linear from hidden_dim to vocab_size
    lm_head = linear_flops(batch_size, seq_len, hidden_dim, vocab_size)

    forward_flops = num_layers * block + lm_head

    # Training: forward + backward (backward ~ 2x forward)
    return 3 * forward_flops
