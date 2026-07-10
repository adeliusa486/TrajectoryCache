"""
Learning-based cache replacement policy (Linear Q-Learning / TD Function Approximation).

Implements a lightweight reinforcement learning baseline where each content item
is represented by state features (normalized spatial urgency and popularity),
and the policy learns expected retention utility Q(s) via temporal-difference updates.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import defaultdict, deque
from typing import Any

from .base import BaseCache, CacheItem

logger = logging.getLogger(__name__)


class QLearningCache(BaseCache):
    """
    QLearningCache - Learning-based cache replacement baseline using linear TD learning.

    Parameters
    ----------
    capacity : int
        Maximum number of items in the cache.
    lr : float
        Learning rate alpha for temporal difference value updates.
    pop_window : float
        Sliding window for popularity estimation (s).
    t_pred : float
        Lookahead horizon for spatial urgency (s).
    alpha_d : float
        Urgency decay constant (s^-1).
    r_rel : float
        Relevance radius in metres.
    """

    name: str = "QLearning"

    def __init__(
        self,
        capacity: int,
        lr: float = 0.01,
        pop_window: float = 300.0,
        t_pred: float = 30.0,
        alpha_d: float = 0.1,
        r_rel: float = 800.0,
    ) -> None:
        super().__init__(capacity)
        self.lr = lr
        self.pop_window = pop_window
        self.t_pred = t_pred
        self.alpha_d = alpha_d
        self.r_rel = r_rel

        # Popularity tracking
        self._req_times: dict[int, deque] = defaultdict(deque)
        self._lock = threading.Lock()

        # Learned weight vector w = [w_u, w_p, w_bias] for Q(f) = w_u*U(f) + w_p*P(f) + w_bias
        self.w_u = 0.30
        self.w_p = 0.70
        self.w_bias = 0.0

        # Tracking training overhead / inference latency
        self.total_inference_time = 0.0
        self.total_training_updates = 0

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
            # Fast TD reward update for hit (+1.0)
            u_norm, p_norm = self._fast_item_features(item_id, item_location, vehicles, catalog, current_time)
            q_val = self.w_u * u_norm + self.w_p * p_norm + self.w_bias
            td_error = 1.0 - q_val
            self.w_u += self.lr * td_error * u_norm
            self.w_p += self.lr * td_error * p_norm
            self.w_bias += self.lr * td_error * 0.1
            # Clip weights to sensible non-negative bounds
            self.w_u = max(0.01, min(1.0, self.w_u))
            self.w_p = max(0.01, min(1.0, self.w_p))
            self.total_training_updates += 1
            return True

        self._misses += 1
        self._fetch_from_backhaul(item_id, item_location, current_time, vehicles, catalog)
        return False

    def _fetch_from_backhaul(
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

        extended_catalog = dict(catalog)
        extended_catalog[new_id] = new_loc

        # Batch compute features exactly like TrajectoryCache._score_all for O(N) speed
        t0 = time.perf_counter()
        raw_urgency = {
            fid: self._raw_urgency(loc, vehicles, t) for fid, loc in extended_catalog.items()
        }
        max_u = max(raw_urgency.values(), default=0.0)
        min_u = min(raw_urgency.values(), default=0.0)
        denom_u = (max_u - min_u) if (max_u - min_u) > 1e-6 else 1.0

        with self._lock:
            pop_counts = {fid: len(self._req_times[fid]) for fid in extended_catalog}
            max_p = max(max((len(dq) for dq in self._req_times.values()), default=1), 1)

        q_vals: dict[int, float] = {}
        for fid in extended_catalog:
            u_norm = (raw_urgency[fid] - min_u) / denom_u
            p_norm = pop_counts[fid] / max_p
            q_vals[fid] = self.w_u * u_norm + self.w_p * p_norm + self.w_bias

        victim_id = min(self._cache, key=lambda i: q_vals.get(i, 0.0))
        val_victim = q_vals.get(victim_id, 0.0)
        val_new = q_vals.get(new_id, 0.0)
        t1 = time.perf_counter()
        self.total_inference_time += (t1 - t0)

        if val_victim < val_new:
            del self._cache[victim_id]
            self._cache[new_id] = CacheItem(item_id=new_id, location=new_loc, timestamp=t)

    def _fast_item_features(
        self,
        fid: int,
        loc: float,
        vehicles: list[dict],
        catalog: dict[int, float],
        t: float,
    ) -> tuple[float, float]:
        raw_u = self._raw_urgency(loc, vehicles, t)
        u_norm = min(1.0, max(0.0, raw_u))
        with self._lock:
            cnt = len(self._req_times.get(fid, []))
            max_cnt = max(1, max((len(dq) for dq in self._req_times.values()), default=1))
        return (u_norm, cnt / max_cnt)

    def _raw_urgency(self, item_location: float, vehicles: list[dict], t: float) -> float:
        if not vehicles:
            return 0.0
        total_urgency = 0.0
        for veh in vehicles:
            d_t0 = abs(item_location - veh["x"])
            speed = veh.get("speed", 0.0)
            direction = veh.get("direction", 1)
            pred_x = veh["x"] + direction * speed * self.t_pred
            d_pred = abs(item_location - pred_x)
            if d_pred <= self.r_rel and d_pred < d_t0:
                approaching_speed = max(speed, 1.0)
                tau = d_pred / approaching_speed
                total_urgency += math.exp(-self.alpha_d * tau)
        return total_urgency

    def _prune_request_window(self, current_time: float) -> None:
        cutoff = current_time - self.pop_window
        for dq in self._req_times.values():
            while dq and dq[0] < cutoff:
                dq.popleft()

    def reset(self) -> None:
        super().reset()
        self._req_times.clear()
        self.total_inference_time = 0.0
        self.total_training_updates = 0
