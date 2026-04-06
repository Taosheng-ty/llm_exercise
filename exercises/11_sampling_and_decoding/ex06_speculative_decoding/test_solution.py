"""Tests for Exercise 06: Speculative Decoding"""

import importlib.util
import os

import pytest
import torch

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
speculative_decode = _mod.speculative_decode


def make_identical_models(vocab_size=10):
    """When draft and target agree perfectly, all tokens should be accepted."""
    fixed_logits = torch.zeros(vocab_size)
    fixed_logits[3] = 10.0  # strongly prefer token 3

    def draft_model(tokens):
        return fixed_logits.clone()

    def target_model(tokens):
        seq_len = len(tokens)
        return fixed_logits.unsqueeze(0).expand(seq_len, -1).clone()

    return draft_model, target_model


def make_disagreeing_models(vocab_size=10):
    """Draft prefers token 3, target prefers token 7 — should reject often."""
    draft_logits = torch.zeros(vocab_size)
    draft_logits[3] = 10.0

    target_logits = torch.zeros(vocab_size)
    target_logits[7] = 10.0

    def draft_model(tokens):
        return draft_logits.clone()

    def target_model(tokens):
        seq_len = len(tokens)
        return target_logits.unsqueeze(0).expand(seq_len, -1).clone()

    return draft_model, target_model


class TestSpeculativeDecoding:
    def test_identical_models_accept_all(self):
        """When draft == target, all tokens should be accepted."""
        draft, target = make_identical_models()
        new_tokens, stats = speculative_decode(
            prefix=[0], draft_model=draft, target_model=target, k=5, random_seed=42
        )
        assert stats["draft_tokens"] == 5
        assert stats["accepted_tokens"] == 5
        # All accepted + bonus token
        assert len(new_tokens) == 6

    def test_disagreeing_models_reject(self):
        """When models disagree strongly, tokens should be rejected."""
        draft, target = make_disagreeing_models()
        new_tokens, stats = speculative_decode(
            prefix=[0], draft_model=draft, target_model=target, k=5, random_seed=42
        )
        assert stats["draft_tokens"] == 5
        # Should reject at least some tokens
        assert stats["accepted_tokens"] < 5
        # Should have at least 1 token (the resampled one)
        assert len(new_tokens) >= 1

    def test_returns_at_least_one_token(self):
        """Should always produce at least one new token."""
        draft, target = make_disagreeing_models()
        new_tokens, stats = speculative_decode(
            prefix=[0, 1, 2], draft_model=draft, target_model=target, k=3, random_seed=0
        )
        assert len(new_tokens) >= 1

    def test_stats_keys(self):
        """Stats dict should contain required keys."""
        draft, target = make_identical_models()
        _, stats = speculative_decode(
            prefix=[0], draft_model=draft, target_model=target, k=3
        )
        assert "draft_tokens" in stats
        assert "accepted_tokens" in stats
        assert stats["draft_tokens"] == 3

    def test_accepted_leq_draft(self):
        """Number of accepted tokens should be <= number of draft tokens."""
        draft, target = make_disagreeing_models()
        for seed in range(10):
            _, stats = speculative_decode(
                prefix=[0], draft_model=draft, target_model=target, k=4, random_seed=seed
            )
            assert stats["accepted_tokens"] <= stats["draft_tokens"]

    def test_k_one(self):
        """k=1 should draft a single token."""
        draft, target = make_identical_models()
        new_tokens, stats = speculative_decode(
            prefix=[0], draft_model=draft, target_model=target, k=1, random_seed=42
        )
        assert stats["draft_tokens"] == 1
        assert len(new_tokens) >= 1

    def test_deterministic_with_seed(self):
        """Same seed should produce same results."""
        draft, target = make_disagreeing_models()
        result1 = speculative_decode(
            prefix=[0, 1], draft_model=draft, target_model=target, k=3, random_seed=123
        )
        result2 = speculative_decode(
            prefix=[0, 1], draft_model=draft, target_model=target, k=3, random_seed=123
        )
        assert result1[0] == result2[0]
        assert result1[1] == result2[1]

    def test_bonus_token_on_full_accept(self):
        """When all k tokens accepted, should get k+1 total new tokens (bonus)."""
        draft, target = make_identical_models()
        new_tokens, stats = speculative_decode(
            prefix=[0], draft_model=draft, target_model=target, k=3, random_seed=42
        )
        if stats["accepted_tokens"] == 3:
            assert len(new_tokens) == 4  # 3 accepted + 1 bonus
