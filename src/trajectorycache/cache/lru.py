"""Least-Recently-Used (LRU) cache baseline."""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

from .base import BaseCache, CacheItem


class LRUCache(BaseCache):
    """Standard LRU eviction policy used as a baseline."""

    name: str = "LRU"

    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)
        self._order: OrderedDict[int, None] = OrderedDict()

    def request(
        self,
        item_id: int,
        item_location: float,
        current_time: float,
        **kwargs,
    ) -> bool:
        if item_id in self._cache:
            self._hits += 1
            self._order.move_to_end(item_id)
            self._cache[item_id].access_count += 1
            self._cache[item_id].timestamp = current_time
            return True

        self._misses += 1

        if len(self._cache) >= self.capacity:
            victim, _ = self._order.popitem(last=False)
            del self._cache[victim]

        self._cache[item_id] = CacheItem(
            item_id=item_id, location=item_location, timestamp=current_time
        )
        self._order[item_id] = None
        return False

    def clear(self) -> None:
        super().clear()
        self._order.clear()
