"""
Solution for Exercise 01: Rollout Data Source with Epoch Tracking
"""

from dataclasses import dataclass, field
import numpy as np
from typing import List, Any


@dataclass
class RolloutDataSource:
    """A data source that serves batches and tracks epochs."""
    data: List[Any] = field(default_factory=list)
    seed: int = 42
    epoch: int = 0
    offset: int = 0

    def load_data(self, items: List[Any]) -> None:
        self.data = list(items)
        self.epoch = 0
        self.offset = 0
        self._shuffle()

    def _shuffle(self) -> None:
        rng = np.random.RandomState(self.seed + self.epoch)
        rng.shuffle(self.data)

    def get_batch(self, batch_size: int) -> List[Any]:
        if len(self.data) == 0:
            raise ValueError("Data source is empty.")
        if batch_size > len(self.data):
            raise ValueError(
                f"batch_size ({batch_size}) exceeds data length ({len(self.data)})."
            )

        if self.offset + batch_size <= len(self.data):
            batch = self.data[self.offset : self.offset + batch_size]
            self.offset += batch_size
        else:
            # Take remainder from current epoch
            batch = self.data[self.offset:]
            remaining = batch_size - len(batch)
            # Advance epoch and reshuffle
            self.epoch += 1
            self._shuffle()
            batch = batch + self.data[:remaining]
            self.offset = remaining

        return batch

    def reset(self) -> None:
        self.epoch = 0
        self.offset = 0
        self._shuffle()

    @property
    def current_epoch(self) -> int:
        return self.epoch

    @property
    def remaining_in_epoch(self) -> int:
        return len(self.data) - self.offset
