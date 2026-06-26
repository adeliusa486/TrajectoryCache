"""LFU, Random, and FIFO eviction baseline caches."""

from __future__ import annotations

from collections import defaultdict, deque

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

    def __init__(self, capacity: int, seed: int | None = None) -> None:
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


class ProximityCache(BaseCache):
    """
    Pure trajectory/proximity-aware eviction policy -- NO popularity term.

    Represents the family of distance/ETA-greedy mobility-aware caching
    schemes in the vehicular edge-caching literature: it evicts the cached
    item with the largest current minimum time-to-encounter (TTE) to any
    vehicle predicted to reach it within the lookahead horizon. Items with
    no approaching vehicle are evicted first (TTE = +inf). This is the
    natural counterpart to LFU's "popularity alone" extreme of
    TrajectoryCache's W=0/W=1 spectrum, used to test whether spatial
    urgency alone (without the popularity blend) is sufficient.
    """

    name: str = "Proximity"

    def __init__(self, capacity: int, t_pred: float = 30.0, r_rel: float = 800.0) -> None:
        super().__init__(capacity)
        self.t_pred = t_pred
        self.r_rel = r_rel

    def request(
        self,
        item_id: int,
        item_location: float,
        current_time: float,
        vehicles: list | None = None,
        catalog: dict | None = None,
        **kwargs,
    ) -> bool:
        vehicles = vehicles or []
        catalog = catalog or {}

        if item_id in self._cache:
            self._hits += 1
            return True

        self._misses += 1

        if len(self._cache) >= self.capacity:
            tte_new = self._min_tte(item_location, vehicles)
            tte_cached = {
                fid: self._min_tte(catalog.get(fid, self._cache[fid].location), vehicles)
                for fid in self._cache
            }
            victim_id = max(tte_cached, key=tte_cached.get)
            if tte_cached[victim_id] > tte_new:
                del self._cache[victim_id]
            else:
                # New item is less urgent than every cached item -- discard it.
                return False

        self._cache[item_id] = CacheItem(
            item_id=item_id, location=item_location, timestamp=current_time
        )
        return False

    def _min_tte(self, loc: float, vehicles: list) -> float:
        best = float("inf")
        for veh in vehicles:
            x_v = veh["x"]
            speed = veh.get("speed", 0.0)
            direction = veh.get("direction", 1.0)
            if speed <= 0:
                continue
            x_hat = x_v + speed * direction * self.t_pred
            if abs(x_hat - loc) > self.r_rel:
                continue
            tte = abs(loc - x_v) / speed
            if tte < best:
                best = tte
        return best


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
