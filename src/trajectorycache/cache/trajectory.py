"""
TrajectoryCache: Spatial-urgency-aware cache replacement heuristic.

Implements the composite scoring function from the paper:
    Score(f) = W * Urgency(f) + (1 - W) * Popularity(f)

Where spatial urgency is derived from real-time vehicle kinematics and
popularity is a sliding-window request count normalized across cache items.
"""

from __future__ import annotations

import logging
import time
import threading
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple, Any

from .base import BaseCache, CacheItem

logger = logging.getLogger(__name__)

_EPSILON = 1e-6  # numerical safety constant


class TrajectoryCache(BaseCache):
    """
    TrajectoryCache (TC) - mobility-aware edge cache replacement.

    Parameters
    ----------
    capacity : int
        Maximum number of items in the cache (C_max).
    urgency_weight : float
        W  [0, 1]. Weight assigned to the spatial-urgency component.
        W = 0 -> pure normalized-LFU.  W = 1 -> pure urgency-driven.
    pop_window : float
        Sliding time-window duration in seconds for popularity counting (W_pop).
    t_pred : float
        Linear lookahead horizon in seconds for vehicle position extrapolation.
    alpha_d : float
        Urgency decay constant in s-1 controlling how steeply urgency falls
        off with increasing time-to-encounter.
    r_rel : float
        Relevance radius in metres. A vehicle's predicted position must fall
        within this distance of an item's geographic location to contribute.
    """

    name: str = "TrajectoryCache"

    def __init__(
        self,
        capacity: int,
        urgency_weight: float = 0.2,
        pop_window: float = 300.0,
        t_pred: float = 30.0,
        alpha_d: float = 0.1,
        r_rel: float = 800.0,
    ) -> None:
        super().__init__(capacity)

        if not 0.0 <= urgency_weight <= 1.0:
            raise ValueError(f"urgency_weight must be in [0, 1]; got {urgency_weight}")

        self.W = urgency_weight
        self.pop_window = pop_window
        self.t_pred = t_pred
        self.alpha_d = alpha_d
        self.r_rel = r_rel

        # Popularity: {item_id -> deque of request timestamps}
        self._req_times: Dict[int, deque] = defaultdict(deque)
        self._lock = threading.Lock()

        # Step-level memoization for urgency
        self._last_t = -1.0
        self._urgency_cache: Dict[float, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def request(
        self,
        item_id: int,
        item_location: float,
        current_time: float,
        vehicles: Optional[List[dict]] = None,
        catalog: Optional[Dict[int, float]] = None,
        **kwargs: Any,
    ) -> bool:
        """
        Process a content request.

        Parameters
        ----------
        item_id : int
            Identifier of the requested content item.
        item_location : float
            Geographic position of the content item along the highway (metres).
        current_time : float
            Wall-clock simulation time in seconds.
        vehicles : list of dicts, optional
            Each dict must contain keys: ``x`` (position m), ``speed`` (m/s),
            ``direction`` (+1 or -1).
        catalog : dict, optional
            {item_id: location} mapping for all items in the cache (needed only
            to compute urgency for cached items - a shallow copy is fine).

        Returns
        -------
        bool
            True on cache hit, False on cache miss.
        """
        vehicles = vehicles or []
        catalog = catalog or {}

        with self._lock:
            self._prune_request_window(current_time)

            # Record request for popularity
            self._req_times[item_id].append(current_time)

        if item_id in self._cache:
            # ---- HIT ----
            self._hits += 1
            return True

        # ---- MISS ----
        self._misses += 1
        self._fetch_from_backhaul(item_id, item_location, current_time, vehicles, catalog)
        return False

    def evict(self) -> Optional[int]:
        """Force-evict the lowest-scoring cached item (used by tests)."""
        if not self._cache:
            return None
        scores = self._score_all_cached([], current_time=time.time())
        victim = min(scores, key=lambda k: scores[k])
        del self._cache[victim]
        logger.debug("Force-evicted item %d", victim)
        return victim

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_from_backhaul(
        self,
        new_id: int,
        new_loc: float,
        t: float,
        vehicles: List[dict],
        catalog: Dict[int, float],
    ) -> None:
        """Insert the newly fetched item, evicting if cache is full."""
        if len(self._cache) < self.capacity:
            self._cache[new_id] = CacheItem(item_id=new_id, location=new_loc, timestamp=t)
            return

        # Compute composite scores for all cached items plus the new item (set C+).
        # Per the paper algorithm: evict the lowest-scoring cached item ONLY if
        # the new item scores higher. Otherwise discard the new item.
        extended_catalog = dict(catalog)
        extended_catalog[new_id] = new_loc
        scores = self._score_all(extended_catalog, vehicles, t)

        victim_id = min(
            self._cache,
            key=lambda i: scores.get(i, 0.0),
        )
        score_victim = scores.get(victim_id, 0.0)
        score_new = scores.get(new_id, 0.0)

        if score_victim < score_new:
            del self._cache[victim_id]
            self._cache[new_id] = CacheItem(item_id=new_id, location=new_loc, timestamp=t)
            logger.debug(
                "Evicted item %d (score=%.4f) -> cached item %d (score=%.4f)",
                victim_id,
                score_victim,
                new_id,
                score_new,
            )
        else:
            # New item is less spatially/historically valuable than all cached items.
            # Discard it without caching (key advantage over LFU: avoids polluting
            # the cache with items that have no urgency AND low popularity).
            logger.debug(
                "Discarded new item %d (score=%.4f); victim %d retained",
                new_id,
                score_new,
                victim_id,
            )

    def _score_all(
        self,
        catalog: Dict[int, float],
        vehicles: List[dict],
        t: float,
    ) -> Dict[int, float]:
        """
        Compute composite Score(f) for every item in *catalog* (= C+).

        Score(f) = W * Urgency(f) + (1 - W) * Popularity(f)

        Popularity is normalized against the global request maximum across ALL
        tracked items (not just C+). This ensures new items are not unfairly
        penalized against cached items that have accumulated counts while being
        served, which would cause TC to always discard new items.
        """
        # --- Spatial urgency ---
        raw_urgency = {
            fid: self._raw_urgency(loc, vehicles, t) for fid, loc in catalog.items()
        }
        urgency_norm = self._min_max_normalize(raw_urgency)

        # --- Historical popularity (globally normalized) ---
        with self._lock:
            pop_counts = {fid: len(self._req_times[fid]) for fid in catalog}
            # Use the global maximum across ALL tracked items so that new items
            # are judged against the same baseline as cached items.
            global_max = max(
                (len(dq) for dq in self._req_times.values()),
                default=1,
            )
        denom = global_max + _EPSILON
        pop_norm = {fid: pop_counts[fid] / denom for fid in catalog}

        # --- Composite score ---
        scores: Dict[int, float] = {}
        for fid in catalog:
            scores[fid] = self.W * urgency_norm[fid] + (1.0 - self.W) * pop_norm[fid]
        return scores

    def _score_all_cached(self, vehicles: List[dict], current_time: float = 0.0) -> Dict[int, float]:
        """Score only items currently in the cache (no new item in C+)."""
        catalog = {fid: item.location for fid, item in self._cache.items()}
        return self._score_all(catalog, vehicles, current_time)

    def _raw_urgency(self, item_loc: float, vehicles: List[dict], t: float) -> float:
        """
        U_raw(f) = Sum u(v, f)  for vehicles whose predicted position falls
        within r_rel of item_loc.

        u(v, f) = 1 / (1 + alpha_d * TTE(v, f))
        TTE(v, f) = |l_f - x_v| / s_v
        xx_hat_v = x_v + s_v * d_v * T_pred
        """
        if t != self._last_t:
            self._last_t = t
            self._urgency_cache.clear()

        if item_loc in self._urgency_cache:
            return self._urgency_cache[item_loc]

        total = 0.0
        for veh in vehicles:
            x_v = veh["x"]
            speed = veh.get("speed", 0.0)
            direction = veh.get("direction", 1.0)

            if speed <= 0:
                continue

            x_hat = x_v + speed * direction * self.t_pred
            if abs(x_hat - item_loc) > self.r_rel:
                continue

            tte = abs(item_loc - x_v) / speed
            u = 1.0 / (1.0 + self.alpha_d * tte)
            total += u

        self._urgency_cache[item_loc] = total
        return total

    @staticmethod
    def _min_max_normalize(raw: Dict[int, float]) -> Dict[int, float]:
        """Min-max normalize a dict of raw scores into [0, 1]."""
        if not raw:
            return {}
        lo = min(raw.values())
        hi = max(raw.values())
        span = hi - lo + _EPSILON
        return {k: (v - lo) / span for k, v in raw.items()}

    def _prune_request_window(self, current_time: float) -> None:
        """Remove request timestamps older than pop_window from all queues."""
        cutoff = current_time - self.pop_window
        for dq in self._req_times.values():
            while dq and dq[0] < cutoff:
                dq.popleft()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_scores(
        self,
        vehicles: List[dict],
        t: float,
        new_item: Optional[Tuple[int, float]] = None,
    ) -> Dict[int, float]:
        """
        Return current composite scores for all cached items.
        Optionally include a candidate new item (id, location).
        """
        catalog = {fid: item.location for fid, item in self._cache.items()}
        if new_item is not None:
            catalog[new_item[0]] = new_item[1]
        return self._score_all(catalog, vehicles, t)

    def popularity_counts(self) -> Dict[int, int]:
        """Return sliding-window request counts for all tracked items."""
        with self._lock:
            return {k: len(v) for k, v in self._req_times.items() if v}
