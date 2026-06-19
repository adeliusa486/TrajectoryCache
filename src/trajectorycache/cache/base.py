"""Abstract base class for all cache replacement policies."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class CacheItem:
    """Lightweight wrapper holding per-item cache metadata."""

    item_id: int
    location: float       # Geographic position along highway (metres)
    timestamp: float      # Simulation time when item was last inserted/accessed
    access_count: int = 0  # Cumulative access count since insertion


class BaseCache(ABC):
    """
    Abstract base cache with shared hit/miss tracking.

    Subclasses must implement ``request()`` and optionally ``evict()``.
    """

    name: str = "BaseCache"

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"Cache capacity must be positive; got {capacity}")
        self.capacity = capacity
        self._cache: Dict[int, CacheItem] = {}
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def request(
        self, item_id: int, item_location: float, current_time: float, **kwargs
    ) -> bool:
        """
        Process a request for *item_id*.

        Returns True on hit, False on miss.
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def __contains__(self, item_id: int) -> bool:
        return item_id in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    @property
    def is_full(self) -> bool:
        return len(self._cache) >= self.capacity

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def total_requests(self) -> int:
        return self._hits + self._misses

    @property
    def miss_rate(self) -> float:
        total = self.total_requests
        return self._misses / total if total > 0 else 0.0

    @property
    def hit_rate(self) -> float:
        total = self.total_requests
        return self._hits / total if total > 0 else 0.0

    def reset_stats(self) -> None:
        """Reset hit/miss counters without clearing cache contents."""
        self._hits = 0
        self._misses = 0

    def clear(self) -> None:
        """Evict all items and reset stats."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def items(self) -> Dict[int, CacheItem]:
        """Return a view of the current cache contents."""
        return dict(self._cache)

    def summary(self) -> dict:
        """Return a stats summary dict for logging/reporting."""
        return {
            "policy": self.name,
            "capacity": self.capacity,
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "miss_rate": round(self.miss_rate * 100, 4),
            "hit_rate": round(self.hit_rate * 100, 4),
        }
