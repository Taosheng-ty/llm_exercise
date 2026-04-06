import importlib.util
import os
import torch
import math

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

compute_rope_frequencies = _mod.compute_rope_frequencies
apply_rope = _mod.apply_rope


def test_frequency_shapes():
    cos, sin = compute_rope_frequencies(head_dim=16, max_seq_len=32)
    assert cos.shape == (32, 16)
    assert sin.shape == (32, 16)


def test_cos_sin_range():
    cos, sin = compute_rope_frequencies(head_dim=16, max_seq_len=32)
    assert cos.min() >= -1.0 and cos.max() <= 1.0
    assert sin.min() >= -1.0 and sin.max() <= 1.0


def test_position_zero_is_identity():
    """At position 0, sin=0 and cos=1, so RoPE should be identity."""
    cos, sin = compute_rope_frequencies(head_dim=8, max_seq_len=4)
    assert torch.allclose(cos[0], torch.ones(8), atol=1e-5)
    assert torch.allclose(sin[0], torch.zeros(8), atol=1e-5)


def test_apply_rope_shape():
    cos, sin = compute_rope_frequencies(head_dim=16, max_seq_len=32)
    x = torch.randn(2, 4, 8, 16)
    out = apply_rope(x, cos, sin)
    assert out.shape == x.shape


def test_rope_preserves_norm():
    """RoPE is a rotation, so it should preserve vector norms."""
    cos, sin = compute_rope_frequencies(head_dim=16, max_seq_len=32)
    x = torch.randn(1, 1, 8, 16)
    out = apply_rope(x, cos, sin)
    x_norms = torch.norm(x, dim=-1)
    out_norms = torch.norm(out, dim=-1)
    assert torch.allclose(x_norms, out_norms, atol=1e-4)


def test_relative_position_dot_product():
    """
    Key property of RoPE: dot product of rotated q and k depends on
    relative position, not absolute position.
    """
    cos, sin = compute_rope_frequencies(head_dim=8, max_seq_len=64)
    torch.manual_seed(42)
    q_vec = torch.randn(1, 1, 1, 8)
    k_vec = torch.randn(1, 1, 1, 8)

    # Place q at pos 5, k at pos 10 (relative = 5)
    cos5, sin5 = cos[5:6].unsqueeze(0).unsqueeze(0), sin[5:6].unsqueeze(0).unsqueeze(0)
    cos10, sin10 = cos[10:11].unsqueeze(0).unsqueeze(0), sin[10:11].unsqueeze(0).unsqueeze(0)
    q_rot_5 = apply_rope(q_vec, cos[5:6].unsqueeze(0), sin[5:6].unsqueeze(0))
    k_rot_10 = apply_rope(k_vec, cos[10:11].unsqueeze(0), sin[10:11].unsqueeze(0))
    dot1 = (q_rot_5 * k_rot_10).sum()

    # Place q at pos 15, k at pos 20 (same relative = 5)
    q_rot_15 = apply_rope(q_vec, cos[15:16].unsqueeze(0), sin[15:16].unsqueeze(0))
    k_rot_20 = apply_rope(k_vec, cos[20:21].unsqueeze(0), sin[20:21].unsqueeze(0))
    dot2 = (q_rot_15 * k_rot_20).sum()

    assert torch.allclose(dot1, dot2, atol=1e-4), (
        f"Dot products should match for same relative position: {dot1.item()} vs {dot2.item()}"
    )
