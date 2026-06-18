"""LFU and Random eviction baseline caches."""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Optional

from .base import BaseCache, CacheItem


class LFUCache(BaseCache):
    """Least-Frequently-Used eviction policy."""

    name: str = "LFU"

    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)
        self._freq: dict[int, int] = defaultdict(int)

    def request(
        self,
        item_id: int,
        item_location: float,
        current_time: float,
        **kwargs,
    ) -> bool:
        self._freq[item_id] += 1

        if item_id in self._cache:
            self._hits += 1
            self._cache[item_id].access_count += 1
            self._cache[item_id].timestamp = current_time
            return True

        self._misses += 1

        if len(self._cache) >= self.capacity:
            # Evict item with minimum frequency (ties broken by insertion order)
            victim = min(self._cache, key=lambda i: self._freq[i])
            del self._cache[victim]

        self._cache[item_id] = CacheItem(
            item_id=item_id, location=item_location, timestamp=current_time
        )
        return False

    def clear(self) -> None:
        super().clear()
        self._freq.clear()


class RandomCache(BaseCache):
    """Random eviction policy."""

    name: str = "Random"

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
            victim = random.choice(list(self._cache.keys()))
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
