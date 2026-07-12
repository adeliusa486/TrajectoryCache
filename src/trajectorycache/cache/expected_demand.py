"""
ExpectedDemandCache (EDC): demand-forecast cache replacement.

Motivation (SUMO diagnosis, 2026-07): the original TrajectoryCache blends a
normalized urgency term additively with popularity, Score = W*U + (1-W)*P.
Instrumentation on real SUMO Krauss traffic showed the urgency term carries
almost no information about near-term requests (Spearman rho ~= 0.05 vs
popularity's 0.5-0.8), so any W > 0 perturbs a strong ranking with noise and
loses to sliding-window LFU. Two design flaws cause this:

  1. Geometric misalignment: urgency extrapolates positions T_pred ahead and
     accepts items within r_rel of the *predicted* point, crediting items the
     vehicle has not yet reached AND items it will already have passed, while
     vehicles actually request items in their forward request window now.
  2. Wrong functional form: expected near-term demand is a *product* --
     (how often is f requested when someone can request it) x (how many
     vehicles are currently positioned to request it) -- not a weighted sum
     of two normalized ranks.

EDC scores every item by an estimate of its demand over the next few tens of
seconds:

    Score(f) = cnt_window(f) * (1 + exposure(f))

    exposure(f) = sum over vehicles v with f inside v's forward request
                  window (0 < (l_f - x_v) * d_v <= r_req) of a TTE-decay
                  weight  1 / (1 + alpha_d * TTE(v, f))   [or 1 if unweighted]

Properties:
  * No blend weight W. When exposure is spatially uniform (concentrated
    content, saturated traffic) the factor (1 + exposure) is ~constant across
    items and the ranking degrades gracefully to sliding-window LFU.
  * When content is geographically dispersed and traffic is sparse, exposure
    varies strongly across items (many items have zero approaching vehicles)
    and the product concentrates cache capacity on items that are both
    popular and physically reachable soon.
  * The +1 popularity floor keeps globally popular items rankable during
    momentary exposure gaps instead of zeroing their score.

Eviction rule matches TrajectoryCache: on a miss with a full cache, evict the
lowest-scoring cached item only if the new item scores strictly higher;
otherwise the new item is not cached.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict, deque
from typing import Any

from .base import BaseCache, CacheItem

logger = logging.getLogger(__name__)


class ExpectedDemandCache(BaseCache):
    """
    Expected-demand cache replacement (multiplicative popularity x exposure).

    Parameters
    ----------
    capacity : int
        Maximum number of items in the cache.
    pop_window : float
        Sliding time-window (s) for popularity counting.
    r_req : float
        Forward request-window length in metres; must match the demand
        model's look-ahead distance (physical parameter, not a tuned blend).
    alpha_d : float
        TTE decay constant (s^-1) used when tte_weight is True.
    tte_weight : bool
        If True, weight each contributing vehicle by 1/(1 + alpha_d * TTE);
        if False (default), count contributing vehicles equally.
    damping : str
        Exposure damping. "none" (default) scores with (1 + exposure).
        "log" scores with (1 + log1p(exposure)); tested on fresh seeds
        (11111..55555, 2026-07-12) and found slightly WORSE than raw in
        every condition -- kept only for reproducibility of that check.
        The corridor-dense regime where EDC trails LFU is a myopia limit
        (current exposure predicts the next ~30 s, but cached items earn
        hits over minutes of residence), not a damping problem.
    """

    name: str = "ExpectedDemand"

    def __init__(
        self,
        capacity: int,
        pop_window: float = 300.0,
        r_req: float = 800.0,
        alpha_d: float = 0.1,
        tte_weight: bool = False,
        damping: str = "none",
    ) -> None:
        super().__init__(capacity)
        if damping not in ("log", "none"):
            raise ValueError(f"damping must be 'log' or 'none'; got {damping!r}")
        self.pop_window = pop_window
        self.r_req = r_req
        self.alpha_d = alpha_d
        self.tte_weight = tte_weight
        self.damping = damping

        self._req_times: dict[int, deque] = defaultdict(deque)
        self._lock = threading.Lock()

        # Step-level memoization of exposure by item location
        self._last_t = -1.0
        self._exposure_cache: dict[float, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def request(
        self,
        item_id: int,
        item_location: float,
        current_time: float,
        vehicles: list[dict] | None = None,
        catalog: dict[int, float] | None = None,
        **kwargs: Any,
    ) -> bool:
        vehicles = vehicles or []
        catalog = catalog or {}

        with self._lock:
            self._prune_request_window(current_time)
            self._req_times[item_id].append(current_time)

        if item_id in self._cache:
            self._hits += 1
            return True

        self._misses += 1
        self._insert_or_discard(item_id, item_location, current_time, vehicles, catalog)
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _insert_or_discard(
        self,
        new_id: int,
        new_loc: float,
        t: float,
        vehicles: list[dict],
        catalog: dict[int, float],
    ) -> None:
        if len(self._cache) < self.capacity:
            self._cache[new_id] = CacheItem(item_id=new_id, location=new_loc, timestamp=t)
            return

        extended = {fid: item.location for fid, item in self._cache.items()}
        extended[new_id] = new_loc
        scores = self._score_all(extended, vehicles, t)

        victim_id = min(self._cache, key=lambda i: scores.get(i, 0.0))
        if scores.get(victim_id, 0.0) < scores.get(new_id, 0.0):
            del self._cache[victim_id]
            self._cache[new_id] = CacheItem(item_id=new_id, location=new_loc, timestamp=t)

    def _score_all(
        self,
        catalog: dict[int, float],
        vehicles: list[dict],
        t: float,
    ) -> dict[int, float]:
        """Score(f) = cnt_window(f) * (1 + g(exposure(f))) for all f in catalog."""
        import math

        with self._lock:
            counts = {fid: len(self._req_times[fid]) for fid in catalog}
        scores: dict[int, float] = {}
        for fid, loc in catalog.items():
            e = self._exposure(loc, vehicles, t)
            g = math.log1p(e) if self.damping == "log" else e
            scores[fid] = counts[fid] * (1.0 + g)
        return scores

    def _exposure(self, item_loc: float, vehicles: list[dict], t: float) -> float:
        """Vehicles currently positioned to request this item soon."""
        if t != self._last_t:
            self._last_t = t
            self._exposure_cache.clear()
        if item_loc in self._exposure_cache:
            return self._exposure_cache[item_loc]

        total = 0.0
        for veh in vehicles:
            speed = veh.get("speed", 0.0)
            if speed <= 0:
                continue
            d = (item_loc - veh["x"]) * veh.get("direction", 1)
            if 0.0 < d <= self.r_req:
                if self.tte_weight:
                    total += 1.0 / (1.0 + self.alpha_d * d / speed)
                else:
                    total += 1.0

        self._exposure_cache[item_loc] = total
        return total

    def _prune_request_window(self, current_time: float) -> None:
        cutoff = current_time - self.pop_window
        for dq in self._req_times.values():
            while dq and dq[0] < cutoff:
                dq.popleft()
