"""
Solution for Exercise 02: Experience Replay Buffer
"""

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class Experience:
    """A single experience entry."""
    prompt: str
    response: str
    reward: float


class ReplayBuffer:
    """Fixed-capacity replay buffer with optional priority sampling."""

    def __init__(self, capacity: int, seed: int = 42):
        self.capacity = capacity
        self.buffer: List[Experience] = []
        self.rng = np.random.RandomState(seed)

    def add(self, experiences: List[Experience]) -> int:
        self.buffer.extend(experiences)
        evicted = max(0, len(self.buffer) - self.capacity)
        if evicted > 0:
            self.buffer = self.buffer[evicted:]
        return evicted

    def sample(self, batch_size: int, prioritized: bool = False) -> List[Experience]:
        if len(self.buffer) == 0:
            raise ValueError("Cannot sample from an empty buffer.")

        if prioritized:
            rewards = np.array([e.reward for e in self.buffer], dtype=np.float64)
            weights = rewards - rewards.min() + 1e-6
            probs = weights / weights.sum()
            indices = self.rng.choice(len(self.buffer), size=batch_size, replace=True, p=probs)
        else:
            indices = self.rng.choice(len(self.buffer), size=batch_size, replace=True)

        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def is_full(self) -> bool:
        return len(self.buffer) >= self.capacity

    def clear(self) -> None:
        self.buffer.clear()
