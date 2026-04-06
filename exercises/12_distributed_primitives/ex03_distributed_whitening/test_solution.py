"""
Tests for Exercise 03: Distributed Advantage Whitening
"""

import importlib.util
import os

import torch

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

distributed_masked_whiten = _mod.distributed_masked_whiten


def test_basic_whitening():
    """With all-ones mask, result should have mean ~0 and std ~1."""
    torch.manual_seed(42)
    values = [torch.randn(10) for _ in range(3)]
    masks = [torch.ones(10) for _ in range(3)]
    result = distributed_masked_whiten(values, masks)
    all_whitened = torch.cat(result)
    assert abs(all_whitened.mean().item()) < 0.3
    assert abs(all_whitened.std().item() - 1.0) < 0.3


def test_matches_single_worker():
    """With one worker, should match standard whitening."""
    torch.manual_seed(0)
    v = torch.randn(20)
    m = torch.ones(20)
    result = distributed_masked_whiten([v], [m])[0]

    # Manual computation
    mean = v.mean()
    std = v.std()
    expected = (v - mean) / std
    assert torch.allclose(result, expected, atol=1e-5)


def test_global_stats_used():
    """
    Each worker's whitening must use global stats, not local stats.
    Verification: same data split differently should give the same result.
    """
    torch.manual_seed(42)
    all_data = torch.randn(20)
    all_mask = torch.ones(20)

    # Split into 2 workers
    v1, v2 = all_data[:10], all_data[10:]
    m1, m2 = all_mask[:10], all_mask[10:]
    result_split = distributed_masked_whiten([v1, v2], [m1, m2])
    combined_split = torch.cat(result_split)

    # Single worker
    result_single = distributed_masked_whiten([all_data], [all_mask])
    combined_single = result_single[0]

    assert torch.allclose(combined_split, combined_single, atol=1e-5)


def test_masked_elements_excluded():
    """Masked-out elements should not affect global statistics."""
    v1 = torch.tensor([1.0, 100.0, 3.0])
    m1 = torch.tensor([1.0, 0.0, 1.0])  # 100.0 is masked out
    v2 = torch.tensor([2.0, 4.0])
    m2 = torch.tensor([1.0, 1.0])

    result = distributed_masked_whiten([v1, v2], [m1, m2])

    # Global stats should be computed from [1, 3, 2, 4] only
    unmasked = torch.tensor([1.0, 3.0, 2.0, 4.0])
    global_mean = unmasked.mean()
    global_var = unmasked.var()  # Bessel-corrected by default

    expected_v2 = (v2 - global_mean) / torch.sqrt(global_var + 1e-8)
    assert torch.allclose(result[1], expected_v2, atol=1e-5)


def test_shift_mean_false():
    """When shift_mean=False, the global mean is added back."""
    torch.manual_seed(42)
    values = [torch.randn(10) for _ in range(2)]
    masks = [torch.ones(10) for _ in range(2)]

    result_shift = distributed_masked_whiten(values, masks, shift_mean=True)
    result_no_shift = distributed_masked_whiten(values, masks, shift_mean=False)

    # Compute global mean
    all_vals = torch.cat(values)
    global_mean = all_vals.mean()

    # no_shift = shift + global_mean
    for rs, rns in zip(result_shift, result_no_shift):
        assert torch.allclose(rns, rs + global_mean, atol=1e-5)


def test_bessel_correction():
    """With only 2 unmasked values, Bessel's correction should apply (N/(N-1) = 2)."""
    v1 = torch.tensor([0.0])
    m1 = torch.tensor([1.0])
    v2 = torch.tensor([2.0])
    m2 = torch.tensor([1.0])

    result = distributed_masked_whiten([v1, v2], [m1, m2])

    # global_mean = 1.0
    # raw_var = (0+4)/2 - 1 = 1.0
    # bessel_var = 1.0 * 2/(2-1) = 2.0
    # whiten(0) = (0 - 1) / sqrt(2 + 1e-8) = -1/sqrt(2)
    expected_v1 = torch.tensor([-1.0 / (2.0 + 1e-8) ** 0.5])
    assert torch.allclose(result[0], expected_v1, atol=1e-5)


def test_zero_mask_raises():
    v = [torch.tensor([1.0, 2.0])]
    m = [torch.tensor([0.0, 0.0])]
    try:
        distributed_masked_whiten(v, m)
        assert False, "Should raise ValueError"
    except ValueError:
        pass
