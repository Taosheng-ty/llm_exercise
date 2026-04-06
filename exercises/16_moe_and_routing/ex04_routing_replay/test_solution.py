import importlib.util
import os

import torch

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

RoutingReplayCache = _mod.RoutingReplayCache
compute_topk_with_replay = _mod.compute_topk_with_replay


def test_cache_record_and_replay():
    cache = RoutingReplayCache()
    indices1 = torch.tensor([[0, 1], [2, 3]])
    indices2 = torch.tensor([[1, 0], [3, 2]])

    cache.record(indices1)
    cache.record(indices2)

    # Forward replay
    out1 = cache.replay_forward()
    assert torch.equal(out1, indices1)
    out2 = cache.replay_forward()
    assert torch.equal(out2, indices2)


def test_cache_stores_copy():
    """Modifying the original tensor should not affect the cached version."""
    cache = RoutingReplayCache()
    indices = torch.tensor([[0, 1], [2, 3]])
    cache.record(indices)
    indices[0, 0] = 99
    replayed = cache.replay_forward()
    assert replayed[0, 0].item() == 0, "Cache should store a copy"


def test_backward_replay_independent():
    cache = RoutingReplayCache()
    cache.record(torch.tensor([[0, 1]]))
    cache.record(torch.tensor([[2, 3]]))

    # Forward uses forward_index
    fwd = cache.replay_forward()
    assert torch.equal(fwd, torch.tensor([[0, 1]]))

    # Backward uses backward_index (starts from 0 independently)
    bwd = cache.replay_backward()
    assert torch.equal(bwd, torch.tensor([[0, 1]]))


def test_clear():
    cache = RoutingReplayCache()
    cache.record(torch.tensor([[0, 1]]))
    cache.replay_forward()
    cache.replay_backward()
    cache.clear()
    assert len(cache.top_indices_list) == 0
    assert cache.forward_index == 0
    assert cache.backward_index == 0


def test_clear_forward_only():
    cache = RoutingReplayCache()
    cache.record(torch.tensor([[0, 1]]))
    cache.replay_forward()
    cache.replay_backward()

    cache.clear_forward()
    assert cache.forward_index == 0
    assert cache.backward_index == 1  # backward index unchanged

    # Can replay forward again
    out = cache.replay_forward()
    assert torch.equal(out, torch.tensor([[0, 1]]))


def test_compute_topk_fallthrough():
    scores = torch.tensor([[0.1, 0.9, 0.5, 0.3]])
    cache = RoutingReplayCache()
    probs, indices = compute_topk_with_replay(scores, 2, cache, "fallthrough")

    assert indices.shape == (1, 2)
    assert 1 in indices[0].tolist()  # Expert 1 has highest score
    assert len(cache.top_indices_list) == 0  # Nothing recorded


def test_compute_topk_record_then_replay():
    torch.manual_seed(42)
    scores = torch.randn(4, 8)
    cache = RoutingReplayCache()

    # Record
    probs_rec, indices_rec = compute_topk_with_replay(scores, 2, cache, "record")
    assert len(cache.top_indices_list) == 1

    # Replay forward with DIFFERENT scores
    new_scores = torch.randn(4, 8)
    probs_replay, indices_replay = compute_topk_with_replay(new_scores, 2, cache, "replay_forward")

    # Indices should be the SAME as recorded (not recomputed from new_scores)
    assert torch.equal(indices_replay, indices_rec)

    # But probs should come from NEW scores gathered at recorded indices
    expected_probs = new_scores.gather(1, indices_rec)
    assert torch.allclose(probs_replay, expected_probs)


def test_compute_topk_replay_backward():
    torch.manual_seed(0)
    scores = torch.randn(3, 6)
    cache = RoutingReplayCache()

    # Record
    _, indices_rec = compute_topk_with_replay(scores, 2, cache, "record")

    # Replay backward with different scores
    new_scores = torch.randn(3, 6)
    _, indices_bwd = compute_topk_with_replay(new_scores, 2, cache, "replay_backward")

    assert torch.equal(indices_bwd, indices_rec)


def test_multi_layer_replay():
    """Simulate recording across multiple layers, then replaying them."""
    cache = RoutingReplayCache()
    num_layers = 4
    recorded_indices = []

    # Record phase (simulating forward pass through 4 MoE layers)
    for _ in range(num_layers):
        scores = torch.randn(8, 6)
        _, indices = compute_topk_with_replay(scores, 2, cache, "record")
        recorded_indices.append(indices.clone())

    assert len(cache.top_indices_list) == num_layers

    # Replay forward phase
    for layer in range(num_layers):
        new_scores = torch.randn(8, 6)
        _, indices = compute_topk_with_replay(new_scores, 2, cache, "replay_forward")
        assert torch.equal(indices, recorded_indices[layer])
