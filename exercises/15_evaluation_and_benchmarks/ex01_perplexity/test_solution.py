"""Tests for Exercise 01: Compute Perplexity."""

import importlib.util
import math
import os

import torch
import pytest

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("solution_ex01", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
compute_perplexity = _mod.compute_perplexity
batch_perplexity = _mod.batch_perplexity


class TestComputePerplexity:
    def test_uniform_log_probs(self):
        """Uniform log probs of -log(10) should give PPL=10."""
        log_probs = torch.full((5,), -math.log(10.0))
        mask = torch.ones(5)
        ppl = compute_perplexity(log_probs, mask)
        assert abs(ppl.item() - 10.0) < 1e-4

    def test_perfect_predictions(self):
        """Log prob of 0 (prob=1) should give PPL=1."""
        log_probs = torch.zeros(4)
        mask = torch.ones(4)
        ppl = compute_perplexity(log_probs, mask)
        assert abs(ppl.item() - 1.0) < 1e-5

    def test_masking(self):
        """Only masked-in tokens should contribute."""
        log_probs = torch.tensor([-1.0, -2.0, -100.0])
        mask = torch.tensor([1.0, 1.0, 0.0])
        ppl = compute_perplexity(log_probs, mask)
        expected = math.exp(1.5)  # -(-1 + -2) / 2 = 1.5
        assert abs(ppl.item() - expected) < 1e-4

    def test_empty_mask(self):
        """All-zero mask should return inf."""
        log_probs = torch.tensor([-1.0, -2.0])
        mask = torch.zeros(2)
        ppl = compute_perplexity(log_probs, mask)
        assert ppl.item() == float("inf")

    def test_single_token(self):
        """Single token with log_prob = -3.0 -> PPL = exp(3.0)."""
        log_probs = torch.tensor([-3.0])
        mask = torch.ones(1)
        ppl = compute_perplexity(log_probs, mask)
        assert abs(ppl.item() - math.exp(3.0)) < 1e-4

    def test_known_distribution(self):
        """Known distribution: avg neg log prob = 2.0 -> PPL = exp(2)."""
        log_probs = torch.tensor([-1.0, -2.0, -3.0])
        mask = torch.ones(3)
        ppl = compute_perplexity(log_probs, mask)
        assert abs(ppl.item() - math.exp(2.0)) < 1e-4


class TestBatchPerplexity:
    def test_batch_of_two(self):
        log_probs = torch.tensor([
            [-1.0, -2.0, -3.0],
            [-math.log(10.0), -math.log(10.0), 0.0],
        ])
        mask = torch.tensor([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
        ])
        ppl = batch_perplexity(log_probs, mask)
        assert ppl.shape == (2,)
        assert abs(ppl[0].item() - math.exp(2.0)) < 1e-4
        assert abs(ppl[1].item() - 10.0) < 1e-4

    def test_batch_with_empty_sequence(self):
        log_probs = torch.tensor([
            [-1.0, -1.0],
            [-2.0, -3.0],
        ])
        mask = torch.tensor([
            [1.0, 1.0],
            [0.0, 0.0],
        ])
        ppl = batch_perplexity(log_probs, mask)
        assert abs(ppl[0].item() - math.exp(1.0)) < 1e-4
        assert ppl[1].item() == float("inf")

    def test_single_batch(self):
        log_probs = torch.tensor([[-2.0, -2.0, -2.0]])
        mask = torch.ones(1, 3)
        ppl = batch_perplexity(log_probs, mask)
        assert ppl.shape == (1,)
        assert abs(ppl[0].item() - math.exp(2.0)) < 1e-4
