"""
SimulationRunner: orchestrates highway, catalog, and cache in a single loop.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

from ..cache.base import BaseCache
from ..content.catalog import ContentCatalog
from .highway import HighwaySimulation

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for a single simulation run."""

    # Highway
    road_length: float = 10_000.0
    n_vehicles: int = 600
    dt: float = 1.0
    mean_speed: float = 25.0
    speed_std: float = 5.0
    platoon_size: int = 10  # vehicles per platoon (SUMO Krauss-like clustering)
    platoon_gap: float = 30.0  # max positional spread within platoon (metres)

    # Content
    n_items: int = 200
    active_zone_length: float = 1600.0
    zipf_alpha: float = 0.8
    r_rel: float = 800.0  # relevance radius (m): vehicle look-ahead for requests

    # Simulation
    n_steps: int = 1_000
    warmup_steps: int = 100
    seed: Optional[int] = 42

    # Cache
    cache_capacity: int = 20


@dataclass
class SimulationResult:
    """Aggregated results from one simulation run."""

    policy: str
    total_requests: int
    hits: int
    misses: int
    hit_rate: float
    miss_rate: float
    duration_s: float
    per_step_hit_rate: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "policy": self.policy,
            "total_requests": self.total_requests,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 6),
            "miss_rate": round(self.miss_rate, 6),
            "duration_s": round(self.duration_s, 3),
        }


class SimulationRunner:
    """
    Drive a cache policy through a highway simulation.

    Parameters
    ----------
    cache : BaseCache
        Instantiated cache policy (already configured).
    config : SimulationConfig
        Simulation parameters.
    """

    def __init__(self, cache: BaseCache, config: Optional[SimulationConfig] = None) -> None:
        self.cache = cache
        self.cfg = config or SimulationConfig()

        if self.cfg.seed is not None:
            import random
            import numpy as np

            random.seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)

        self.highway = HighwaySimulation(
            road_length=self.cfg.road_length,
            n_vehicles=self.cfg.n_vehicles,
            dt=self.cfg.dt,
            mean_speed=self.cfg.mean_speed,
            speed_std=self.cfg.speed_std,
            platoon_size=self.cfg.platoon_size,
            platoon_gap=self.cfg.platoon_gap,
            seed=self.cfg.seed,
        )
        self.catalog = ContentCatalog(
            n_items=self.cfg.n_items,
            road_length=self.cfg.road_length,
            active_zone_length=self.cfg.active_zone_length,
            zipf_alpha=self.cfg.zipf_alpha,
            seed=self.cfg.seed,
        )

    # ------------------------------------------------------------------

    def run(self, verbose: bool = False) -> SimulationResult:
        """Execute the full simulation and return aggregated results."""
        self.cache.clear()
        location_map = self.catalog.location_map()
        per_step_hit_rate: List[float] = []
        wall_start = time.perf_counter()

        total_steps = self.cfg.warmup_steps + self.cfg.n_steps

        for step in range(total_steps):
            # Advance highway
            vehicles = self.highway.step()
            t = self.highway.current_time

            # Reset stats after warm-up
            if step == self.cfg.warmup_steps:
                self.cache.reset_stats()

            # Generate content requests driven by vehicle positions.
            # Each vehicle looks ahead by r_request metres and requests the
            # nearest spatially relevant item, weighted by Zipf popularity.
            # This spatial coupling is the core assumption of the paper:
            # vehicles request content that is physically ahead of them.
            requests = self.catalog.generate_vehicle_requests(
                vehicles=vehicles,
                r_request=self.cfg.r_rel,
            )

            n_requests = max(len(requests), 1)  # avoid division by zero
            step_hits = 0
            for item in requests:
                hit = self.cache.request(
                    item_id=item.item_id,
                    item_location=item.location,
                    current_time=t,
                    vehicles=vehicles,
                    catalog=location_map,
                )
                if hit:
                    step_hits += 1

            if step >= self.cfg.warmup_steps and n_requests > 0:
                per_step_hit_rate.append(step_hits / n_requests)

            if verbose and step % 100 == 0:
                logger.info(
                    "Step %d/%d | t=%.1fs | cache hit_rate=%.2f%%",
                    step,
                    total_steps,
                    t,
                    self.cache.hit_rate * 100,
                )

        wall_end = time.perf_counter()
        summary = self.cache.summary()

        return SimulationResult(
            policy=summary["policy"],
            total_requests=summary["hits"] + summary["misses"],
            hits=summary["hits"],
            misses=summary["misses"],
            hit_rate=summary["hit_rate"] / 100.0,
            miss_rate=summary["miss_rate"] / 100.0,
            duration_s=wall_end - wall_start,
            per_step_hit_rate=per_step_hit_rate,
        )
