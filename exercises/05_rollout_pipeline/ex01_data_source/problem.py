"""
Exercise 01: Rollout Data Source with Epoch Tracking

Implement a data source that serves batches from a dataset, automatically
tracking epochs and reshuffling when the data is exhausted.

This is inspired by slime's RolloutDataSource (slime/rollout/data_source.py),
which maintains sample_offset, epoch_id, and reshuffles on epoch boundaries.

Key concepts:
- Epoch tracking: when all data has been served, increment the epoch counter
- Reproducible shuffling: use a seed + epoch_id so each epoch has a
  deterministic but different order
- Wrap-around: if a batch request spans an epoch boundary, serve the remaining
  items from the old epoch and fill the rest from the newly shuffled epoch
"""

from dataclasses import dataclass, field
import numpy as np
from typing import List, Any


@dataclass
class RolloutDataSource:
    """A data source that serves batches and tracks epochs.

    Attributes:
        data: The list of data items loaded from a source.
        seed: Random seed for reproducible shuffling.
        epoch: Current epoch number (starts at 0, increments when data exhausted).
        offset: Current position in the (shuffled) data.
    """
    data: List[Any] = field(default_factory=list)
    seed: int = 42
    epoch: int = 0
    offset: int = 0

    def load_data(self, items: List[Any]) -> None:
        """Load data items into the source and shuffle for the first epoch.

        Args:
            items: List of data items to serve.
        """
        # TODO: Store a copy of items in self.data, reset epoch/offset to 0,
        # and shuffle data for epoch 0 using self._shuffle().
        raise NotImplementedError

    def _shuffle(self) -> None:
        """Shuffle self.data in-place using numpy with seed = self.seed + self.epoch."""
        # TODO: Create a numpy RandomState with (self.seed + self.epoch)
        # and shuffle self.data in-place.
        raise NotImplementedError

    def get_batch(self, batch_size: int) -> List[Any]:
        """Get the next batch of items. Wraps around epoch boundary if needed.

        If the remaining items in the current epoch are fewer than batch_size,
        take whatever is left, increment the epoch, reshuffle, and fill the
        rest from the new epoch.

        Args:
            batch_size: Number of items to return.

        Returns:
            A list of batch_size items.

        Raises:
            ValueError: If data is empty or batch_size > len(data).
        """
        # TODO: Implement batch retrieval with epoch wrap-around.
        raise NotImplementedError

    def reset(self) -> None:
        """Reset to epoch 0, offset 0, and reshuffle."""
        # TODO: Reset epoch and offset, then reshuffle.
        raise NotImplementedError

    @property
    def current_epoch(self) -> int:
        """Return the current epoch number."""
        # TODO
        raise NotImplementedError

    @property
    def remaining_in_epoch(self) -> int:
        """Return how many items are left in the current epoch."""
        # TODO
        raise NotImplementedError
