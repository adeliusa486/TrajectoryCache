"""LFU, Random, and FIFO eviction baseline caches."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Optional

import numpy as np

from .base import BaseCache, CacheItem


class LFUCache(BaseCache):
    """
    Least-Frequently-Used eviction policy with a sliding time window.

    Matches TrajectoryCache's popularity window (pop_window seconds) so that
    the comparison between TC and LFU is fair: both use the same time-horizon
    for frequency counting.
    """

    name: str = "LFU"

    def __init__(self, capacity: int, pop_window: float = 300.0) -> None:
        super().__init__(capacity)
        self.pop_window = pop_window
        # {item_id -> deque of request timestamps within the window}
        self._req_times: dict[int, deque] = defaultdict(deque)

    def _freq(self, item_id: int) -> int:
        """Count of requests in the sliding window."""
        return len(self._req_times[item_id])

    def _prune(self, current_time: float) -> None:
        """Remove timestamps older than pop_window."""
        cutoff = current_time - self.pop_window
        for dq in self._req_times.values():
            while dq and dq[0] < cutoff:
                dq.popleft()

    def request(
        self,
        item_id: int,
        item_location: float,
        current_time: float,
        **kwargs,
    ) -> bool:
        self._prune(current_time)
        self._req_times[item_id].append(current_time)

        if item_id in self._cache:
            self._hits += 1
            self._cache[item_id].access_count += 1
            self._cache[item_id].timestamp = current_time
            return True

        self._misses += 1

        if len(self._cache) >= self.capacity:
            # Evict item with minimum frequency in the sliding window
            victim = min(self._cache, key=lambda i: self._freq(i))
            del self._cache[victim]

        self._cache[item_id] = CacheItem(
            item_id=item_id, location=item_location, timestamp=current_time
        )
        return False

    def clear(self) -> None:
        super().clear()
        self._req_times.clear()


class RandomCache(BaseCache):
    """Random eviction policy (seeded for reproducibility)."""

    name: str = "Random"

    def __init__(self, capacity: int, seed: Optional[int] = None) -> None:
        super().__init__(capacity)
        self._rng = np.random.default_rng(seed)

    def request(
        self,
        item_id: int,
        item_location: float,
        current_time: float,
        **kwargs,
    ) -> bool:
        if item_id in self._cache:
            self._hits += 1
            return True

        self._misses += 1

        if len(self._cache) >= self.capacity:
            keys = list(self._cache.keys())
            victim = keys[int(self._rng.integers(len(keys)))]
            del self._cache[victim]

        self._cache[item_id] = CacheItem(
            item_id=item_id, location=item_location, timestamp=current_time
        )
        return False


class FIFOCache(BaseCache):
    """First-In-First-Out eviction policy."""

    name: str = "FIFO"

    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)
        self._insertion_order: list[int] = []

    def request(
        self,
        item_id: int,
        item_location: float,
        current_time: float,
        **kwargs,
    ) -> bool:
        if item_id in self._cache:
            self._hits += 1
            return True

        self._misses += 1

        if len(self._cache) >= self.capacity:
            victim = self._insertion_order.pop(0)
            del self._cache[victim]

        self._cache[item_id] = CacheItem(
            item_id=item_id, location=item_location, timestamp=current_time
        )
        self._insertion_order.append(item_id)
        return False

    def clear(self) -> None:
        super().clear()
        self._insertion_order.clear()
