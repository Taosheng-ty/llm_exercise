import importlib.util
import os
import torch

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

create_sliding_window_mask = _mod.create_sliding_window_mask
sliding_window_attention = _mod.sliding_window_attention


def test_mask_shape():
    mask = create_sliding_window_mask(8, window_size=3)
    assert mask.shape == (8, 8)
    assert mask.dtype == torch.bool


def test_mask_window_pattern():
    mask = create_sliding_window_mask(6, window_size=2)
    # Token 0: attends to [0] only
    assert not mask[0, 0].item()
    assert mask[0, 1].item()
    # Token 1: attends to [0, 1]
    assert not mask[1, 0].item()
    assert not mask[1, 1].item()
    assert mask[1, 2].item()
    # Token 3: attends to [2, 3]
    assert mask[3, 1].item()  # outside window
    assert not mask[3, 2].item()
    assert not mask[3, 3].item()
    assert mask[3, 4].item()  # future


def test_attention_outside_window_is_zero():
    torch.manual_seed(0)
    B, H, S, D = 1, 1, 8, 16
    Q = torch.randn(B, H, S, D)
    K = torch.randn(B, H, S, D)
    V = torch.randn(B, H, S, D)
    _, attn = sliding_window_attention(Q, K, V, window_size=3)

    # Check that positions outside window get zero weight
    for i in range(S):
        for j in range(S):
            if j > i or j < i - 2:  # outside window or future
                assert attn[0, 0, i, j].item() < 1e-6, (
                    f"Position ({i},{j}) should have zero weight"
                )


def test_attention_weights_sum_to_one():
    B, H, S, D = 2, 2, 8, 16
    Q = torch.randn(B, H, S, D)
    K = torch.randn(B, H, S, D)
    V = torch.randn(B, H, S, D)
    _, attn = sliding_window_attention(Q, K, V, window_size=4)
    sums = attn.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_full_window_equals_causal():
    """When window_size >= seq_len, sliding window = causal attention."""
    import torch.nn.functional as F
    import math
    torch.manual_seed(1)
    B, H, S, D = 1, 2, 6, 8
    Q = torch.randn(B, H, S, D)
    K = torch.randn(B, H, S, D)
    V = torch.randn(B, H, S, D)

    out_sw, _ = sliding_window_attention(Q, K, V, window_size=S)

    # Causal attention
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)
    causal_mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal_mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    out_causal = torch.matmul(attn, V)

    assert torch.allclose(out_sw, out_causal, atol=1e-5)
