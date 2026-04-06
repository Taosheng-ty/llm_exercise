"""Tests for Exercise 01: Learning Rate Scheduler"""

import importlib.util
import os
import math
import torch
import pytest

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
WarmupCosineDecayLR = _mod.WarmupCosineDecayLR
WarmupLinearDecayLR = _mod.WarmupLinearDecayLR


def _make_optimizer(lr=1e-3):
    model = torch.nn.Linear(4, 2)
    return torch.optim.SGD(model.parameters(), lr=lr)


class TestWarmupCosineDecayLR:
    def test_starts_at_zero(self):
        opt = _make_optimizer(lr=1e-3)
        sched = WarmupCosineDecayLR(opt, warmup_steps=10, total_steps=100)
        # At step 0, lr should be 0
        assert sched.get_lr()[0] == pytest.approx(0.0)

    def test_warmup_midpoint(self):
        opt = _make_optimizer(lr=1e-3)
        sched = WarmupCosineDecayLR(opt, warmup_steps=10, total_steps=100)
        for _ in range(5):
            sched.step()
        assert sched.get_lr()[0] == pytest.approx(0.5e-3)

    def test_reaches_base_lr_at_warmup_end(self):
        opt = _make_optimizer(lr=1e-3)
        sched = WarmupCosineDecayLR(opt, warmup_steps=10, total_steps=100)
        for _ in range(10):
            sched.step()
        assert sched.get_lr()[0] == pytest.approx(1e-3, rel=1e-5)

    def test_cosine_midpoint(self):
        opt = _make_optimizer(lr=1e-3)
        sched = WarmupCosineDecayLR(opt, warmup_steps=0, total_steps=100, min_lr=0.0)
        for _ in range(50):
            sched.step()
        # At midpoint of cosine, lr should be base_lr / 2
        assert sched.get_lr()[0] == pytest.approx(0.5e-3, rel=1e-5)

    def test_reaches_min_lr_at_end(self):
        opt = _make_optimizer(lr=1e-3)
        sched = WarmupCosineDecayLR(opt, warmup_steps=10, total_steps=100, min_lr=1e-5)
        for _ in range(100):
            sched.step()
        assert sched.get_lr()[0] == pytest.approx(1e-5, rel=1e-5)

    def test_stays_at_min_lr_after_total_steps(self):
        opt = _make_optimizer(lr=1e-3)
        sched = WarmupCosineDecayLR(opt, warmup_steps=10, total_steps=100, min_lr=1e-5)
        for _ in range(150):
            sched.step()
        assert sched.get_lr()[0] == pytest.approx(1e-5, rel=1e-5)

    def test_monotonic_decrease_after_warmup(self):
        opt = _make_optimizer(lr=1e-3)
        sched = WarmupCosineDecayLR(opt, warmup_steps=10, total_steps=100)
        lrs = []
        for _ in range(100):
            sched.step()
            lrs.append(sched.get_lr()[0])
        # After warmup (step 10+), LRs should be non-increasing
        decay_lrs = lrs[10:]
        for i in range(1, len(decay_lrs)):
            assert decay_lrs[i] <= decay_lrs[i - 1] + 1e-10


class TestWarmupLinearDecayLR:
    def test_starts_at_zero(self):
        opt = _make_optimizer(lr=1e-3)
        sched = WarmupLinearDecayLR(opt, warmup_steps=10, total_steps=100)
        assert sched.get_lr()[0] == pytest.approx(0.0)

    def test_warmup_midpoint(self):
        opt = _make_optimizer(lr=1e-3)
        sched = WarmupLinearDecayLR(opt, warmup_steps=10, total_steps=100)
        for _ in range(5):
            sched.step()
        assert sched.get_lr()[0] == pytest.approx(0.5e-3)

    def test_reaches_base_lr_at_warmup_end(self):
        opt = _make_optimizer(lr=1e-3)
        sched = WarmupLinearDecayLR(opt, warmup_steps=10, total_steps=100)
        for _ in range(10):
            sched.step()
        assert sched.get_lr()[0] == pytest.approx(1e-3, rel=1e-5)

    def test_linear_decay_midpoint(self):
        opt = _make_optimizer(lr=1e-3)
        sched = WarmupLinearDecayLR(opt, warmup_steps=0, total_steps=100, min_lr=0.0)
        for _ in range(50):
            sched.step()
        assert sched.get_lr()[0] == pytest.approx(0.5e-3, rel=1e-5)

    def test_reaches_min_lr_at_end(self):
        opt = _make_optimizer(lr=1e-3)
        sched = WarmupLinearDecayLR(opt, warmup_steps=10, total_steps=100, min_lr=1e-5)
        for _ in range(100):
            sched.step()
        assert sched.get_lr()[0] == pytest.approx(1e-5, rel=1e-4)

    def test_stays_at_min_lr_after_total_steps(self):
        opt = _make_optimizer(lr=1e-3)
        sched = WarmupLinearDecayLR(opt, warmup_steps=10, total_steps=100, min_lr=1e-5)
        for _ in range(150):
            sched.step()
        assert sched.get_lr()[0] == pytest.approx(1e-5, rel=1e-5)
