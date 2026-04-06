"""Tests for Exercise 03: Dynamic Sampling Filters."""

import importlib.util
import os

import pytest

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

Sample = _mod.Sample
FilterOutput = _mod.FilterOutput
filter_identical_rewards = _mod.filter_identical_rewards
filter_low_max_reward = _mod.filter_low_max_reward
filter_short_responses = _mod.filter_short_responses
FilterChain = _mod.FilterChain


def _group(rewards, responses=None):
    """Helper to create a sample group."""
    if responses is None:
        responses = [f"response_{i} padding text here" for i in range(len(rewards))]
    return [
        Sample(prompt="test prompt", response=responses[i], reward=rewards[i])
        for i in range(len(rewards))
    ]


def test_filter_identical_rewards_drop():
    out = filter_identical_rewards(_group([1.0, 1.0, 1.0]))
    assert not out.keep
    assert out.reason is not None and "identical" in out.reason


def test_filter_identical_rewards_keep():
    out = filter_identical_rewards(_group([1.0, 2.0, 1.5]))
    assert out.keep


def test_filter_low_max_reward_drop():
    out = filter_low_max_reward(_group([0.1, 0.2, 0.3]), threshold=0.5)
    assert not out.keep
    assert "low_max" in out.reason


def test_filter_low_max_reward_keep():
    out = filter_low_max_reward(_group([0.1, 0.6]), threshold=0.5)
    assert out.keep


def test_filter_short_responses_drop():
    out = filter_short_responses(
        _group([1.0, 2.0], responses=["hi", "ok"]),
        min_length=10
    )
    assert not out.keep


def test_filter_short_responses_keep_if_any_long():
    out = filter_short_responses(
        _group([1.0, 2.0], responses=["hi", "this is a long enough response"]),
        min_length=10
    )
    assert out.keep


def test_filter_chain_all_pass():
    chain = FilterChain([filter_identical_rewards, filter_short_responses])
    out = chain.apply(_group([1.0, 2.0]))
    assert out.keep


def test_filter_chain_short_circuit():
    chain = FilterChain([filter_identical_rewards, filter_short_responses])
    out = chain.apply(_group([1.0, 1.0, 1.0]))
    assert not out.keep
    assert "identical" in out.reason


def test_filter_chain_drop_reason_counts():
    chain = FilterChain([filter_identical_rewards])
    chain.apply(_group([1.0, 1.0]))
    chain.apply(_group([2.0, 2.0]))
    chain.apply(_group([1.0, 3.0]))  # this passes
    counts = chain.drop_reason_counts
    assert len(counts) == 2  # two different reasons (1.0 and 2.0)
    assert sum(counts.values()) == 2
