"""
Exercise 02: Experience Replay Buffer

Implement a fixed-capacity experience replay buffer commonly used in RL.
When the buffer is full, the oldest samples are evicted (FIFO).
Supports uniform random sampling and optional priority-based sampling
where higher-reward samples are more likely to be drawn.

Inspired by slime's RolloutDataSourceWithBuffer
(slime/rollout/data_source.py) which maintains an internal buffer and
supports custom buffer_filter functions.

Key concepts:
- FIFO eviction: oldest entries are removed when capacity is exceeded
- Uniform sampling: each stored experience is equally likely
- Priority sampling: probability proportional to (reward - min_reward + epsilon)
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class Experience:
    """A single experience entry."""
    prompt: str
    response: str
    reward: float


class ReplayBuffer:
    """Fixed-capacity replay buffer with optional priority sampling.

    Args:
        capacity: Maximum number of experiences to store.
        seed: Random seed for reproducible sampling.
    """

    def __init__(self, capacity: int, seed: int = 42):
        # TODO: Initialize buffer storage, capacity, and RNG.
        # The internal buffer should be stored as self.buffer (a list of Experience).
        raise NotImplementedError

    def add(self, experiences: List[Experience]) -> int:
        """Add experiences to the buffer. Evict oldest if capacity exceeded.

        Args:
            experiences: List of Experience objects to add.

        Returns:
            Number of experiences evicted due to capacity overflow.
        """
        # TODO: Append experiences, then evict oldest entries if over capacity.
        # Return the count of evicted entries.
        raise NotImplementedError

    def sample(self, batch_size: int, prioritized: bool = False) -> List[Experience]:
        """Sample a batch of experiences from the buffer.

        Args:
            batch_size: Number of experiences to sample.
            prioritized: If True, sample with probability proportional to
                         (reward - min_reward + 1e-6). If False, uniform.

        Returns:
            List of sampled Experience objects (with replacement if
            batch_size > len(buffer)).

        Raises:
            ValueError: If the buffer is empty.
        """
        # TODO: Implement uniform and priority-based sampling.
        raise NotImplementedError

    def __len__(self) -> int:
        """Return current number of experiences in the buffer."""
        # TODO
        raise NotImplementedError

    @property
    def is_full(self) -> bool:
        """Return True if the buffer is at capacity."""
        # TODO
        raise NotImplementedError

    def clear(self) -> None:
        """Remove all experiences from the buffer."""
        # TODO
        raise NotImplementedError
