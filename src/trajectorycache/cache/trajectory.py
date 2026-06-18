"""
TrajectoryCache: Spatial-urgency-aware cache replacement heuristic.

Implements the composite scoring function from the paper:
    Score(f) = W * Urgency(f) + (1 - W) * Popularity(f)

Where spatial urgency is derived from real-time vehicle kinematics and
popularity is a sliding-window request count normalized across cache items.
"""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

from .base import BaseCache, CacheItem

logger = logging.getLogger(__name__)

_EPSILON = 1e-6  # numerical safety constant


class TrajectoryCache(BaseCache):
    """
    TrajectoryCache (TC) — mobility-aware edge cache replacement.

    Parameters
    ----------
    capacity : int
        Maximum number of items in the cache (C_max).
    urgency_weight : float
        W ∈ [0, 1]. Weight assigned to the spatial-urgency component.
        W = 0 → pure normalized-LFU.  W = 1 → pure urgency-driven.
    pop_window : float
        Sliding time-window duration in seconds for popularity counting (W_pop).
    t_pred : float
        Linear lookahead horizon in seconds for vehicle position extrapolation.
    alpha_d : float
        Urgency decay constant in s⁻¹ controlling how steeply urgency falls
        off with increasing time-to-encounter.
    r_rel : float
        Relevance radius in metres. A vehicle's predicted position must fall
        within this distance of an item's geographic location to contribute.
    """

    name: str = "TrajectoryCache"

    def __init__(
        self,
        capacity: int,
        urgency_weight: float = 0.1,
        pop_window: float = 300.0,
        t_pred: float = 3.0,
        alpha_d: float = 0.5,
        r_rel: float = 500.0,
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
            to compute urgency for cached items — a shallow copy is fine).

        Returns
        -------
        bool
            True on cache hit, False on cache miss.
        """
        vehicles = vehicles or []
        catalog = catalog or {}

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
        scores = self._score_all_cached({})
        victim = min(scores, key=scores.get)
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

        # Compute scores for all cached items + new item (set C⁺)
        extended_catalog = dict(catalog)
        extended_catalog[new_id] = new_loc

        scores = self._score_all(extended_catalog, vehicles, t)

        victim_id = min(
            (item_id for item_id in self._cache),
            key=lambda i: scores.get(i, 0.0),
        )
        score_victim = scores.get(victim_id, 0.0)
        score_new = scores.get(new_id, 0.0)

        if score_victim < score_new:
            # Evict victim and cache new item
            del self._cache[victim_id]
            self._cache[new_id] = CacheItem(item_id=new_id, location=new_loc, timestamp=t)
            logger.debug(
                "Evicted item %d (score=%.4f) → cached item %d (score=%.4f)",
                victim_id,
                score_victim,
                new_id,
                score_new,
            )
        else:
            # New item is less valuable — discard it
            logger.debug(
                "Discarded new item %d (score=%.4f); victim %d retained (score=%.4f)",
                new_id,
                score_new,
                victim_id,
                score_victim,
            )

    def _score_all(
        self,
        catalog: Dict[int, float],
        vehicles: List[dict],
        t: float,
    ) -> Dict[int, float]:
        """
        Compute composite Score(f) for every item in *catalog* (= C⁺).

        Score(f) = W * Urgency(f) + (1 - W) * Popularity(f)
        """
        # --- Spatial urgency ---
        raw_urgency = {fid: self._raw_urgency(loc, vehicles) for fid, loc in catalog.items()}
        urgency_norm = self._min_max_normalize(raw_urgency)

        # --- Historical popularity ---
        pop_counts = {fid: len(self._req_times[fid]) for fid in catalog}
        pop_norm = self._popularity_normalize(pop_counts)

        # --- Composite score ---
        scores: Dict[int, float] = {}
        for fid in catalog:
            scores[fid] = self.W * urgency_norm[fid] + (1.0 - self.W) * pop_norm[fid]
        return scores

    def _score_all_cached(self, vehicles: dict) -> Dict[int, float]:
        """Score only items currently in the cache (no new item in C⁺)."""
        catalog = {fid: item.location for fid, item in self._cache.items()}
        return self._score_all(catalog, vehicles, 0.0)

    def _raw_urgency(self, item_loc: float, vehicles: List[dict]) -> float:
        """
        U_raw(f) = Σ u(v, f)  for vehicles whose predicted position falls
        within r_rel of item_loc.

        u(v, f) = 1 / (1 + α_d * TTE(v, f))
        TTE(v, f) = |ℓ_f - x_v| / s_v
        x̂_v = x_v + s_v * d_v * T_pred
        """
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

    @staticmethod
    def _popularity_normalize(counts: Dict[int, int]) -> Dict[int, float]:
        """Normalize popularity counts by the maximum count in C⁺."""
        if not counts:
            return {}
        max_count = max(counts.values())
        denom = max_count + _EPSILON
        return {k: v / denom for k, v in counts.items()}

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
        return {k: len(v) for k, v in self._req_times.items() if v}
