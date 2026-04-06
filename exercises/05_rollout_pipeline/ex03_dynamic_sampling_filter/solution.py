"""
Solution for Exercise 03: Dynamic Sampling Filters
"""

from dataclasses import dataclass
from typing import List, Callable, Optional
from collections import defaultdict
import numpy as np


@dataclass
class Sample:
    """A single sample with a response and reward."""
    prompt: str
    response: str
    reward: float


@dataclass
class FilterOutput:
    """Result of a filter decision."""
    keep: bool
    reason: Optional[str] = None


def filter_identical_rewards(samples: List[Sample]) -> FilterOutput:
    rewards = [s.reward for s in samples]
    std = np.std(rewards, dtype=np.float64)
    if std < 1e-8:
        return FilterOutput(keep=False, reason=f"identical_rewards_{rewards[0]:.1f}")
    return FilterOutput(keep=True)


def filter_low_max_reward(samples: List[Sample], threshold: float = 0.5) -> FilterOutput:
    max_reward = max(s.reward for s in samples)
    if max_reward < threshold:
        return FilterOutput(keep=False, reason=f"low_max_reward_{max_reward:.2f}")
    return FilterOutput(keep=True)


def filter_short_responses(samples: List[Sample], min_length: int = 10) -> FilterOutput:
    if all(len(s.response) < min_length for s in samples):
        return FilterOutput(keep=False, reason=f"all_responses_too_short")
    return FilterOutput(keep=True)


class FilterChain:
    """Chain multiple filters together."""

    def __init__(self, filters: List[Callable[[List[Sample]], FilterOutput]]):
        self.filters = filters
        self._drop_reason_counts: dict = defaultdict(int)

    def apply(self, samples: List[Sample]) -> FilterOutput:
        for f in self.filters:
            output = f(samples)
            if not output.keep:
                if output.reason:
                    self._drop_reason_counts[output.reason] += 1
                return output
        return FilterOutput(keep=True)

    @property
    def drop_reason_counts(self) -> dict:
        return dict(self._drop_reason_counts)
