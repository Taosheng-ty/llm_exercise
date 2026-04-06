import importlib.util
import os
import torch
import torch.nn.functional as F

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

create_causal_mask = _mod.create_causal_mask
apply_causal_mask = _mod.apply_causal_mask


def test_mask_shape_2d():
    mask = create_causal_mask(4)
    assert mask.shape == (4, 4)
    assert mask.dtype == torch.bool


def test_mask_shape_broadcastable():
    mask = create_causal_mask(4, batch_broadcastable=True)
    assert mask.shape == (1, 1, 4, 4)


def test_mask_is_upper_triangular():
    mask = create_causal_mask(4)
    # Diagonal and below should be False (not masked)
    for i in range(4):
        for j in range(4):
            if j <= i:
                assert not mask[i, j].item(), f"Position ({i},{j}) should not be masked"
            else:
                assert mask[i, j].item(), f"Position ({i},{j}) should be masked"


def test_apply_mask_sets_neg_inf():
    scores = torch.ones(4, 4)
    mask = create_causal_mask(4)
    masked_scores = apply_causal_mask(scores, mask)
    # Future positions should be -inf
    assert masked_scores[0, 1] == float("-inf")
    assert masked_scores[0, 3] == float("-inf")
    # Current/past positions should be unchanged
    assert masked_scores[0, 0] == 1.0
    assert masked_scores[2, 1] == 1.0


def test_softmax_after_mask():
    scores = torch.randn(1, 1, 4, 4)
    mask = create_causal_mask(4, batch_broadcastable=True)
    masked_scores = apply_causal_mask(scores, mask)
    probs = F.softmax(masked_scores, dim=-1)
    # First row: only attends to position 0
    assert torch.allclose(probs[0, 0, 0, 0], torch.tensor(1.0), atol=1e-5)
    # Future positions get zero probability
    assert probs[0, 0, 0, 1].item() < 1e-6
