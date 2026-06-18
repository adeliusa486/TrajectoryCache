"""
TrajectoryCache — Spatial-urgency-aware edge cache for vehicular networks.

Quick start
-----------
>>> from trajectorycache import TrajectoryCache, SimulationRunner, SimulationConfig
>>> cache = TrajectoryCache(capacity=20, urgency_weight=0.5)
>>> cfg   = SimulationConfig(n_steps=500, seed=42)
>>> runner = SimulationRunner(cache=cache, config=cfg)
>>> result = runner.run()
>>> print(result.hit_rate)
"""
__version__ = "0.1.0"

from .cache import (
    BaseCache,
    CacheItem,
    FIFOCache,
    LFUCache,
    LRUCache,
    RandomCache,
    TrajectoryCache,
    build_cache,
)
from .content import ContentCatalog, ContentItem
from .evaluation import EvalMetrics, compute_metrics, run_benchmark
from .simulation import SimulationConfig, SimulationResult, SimulationRunner

__all__ = [
    # Cache
    "BaseCache",
    "CacheItem",
    "TrajectoryCache",
    "LRUCache",
    "LFUCache",
    "RandomCache",
    "FIFOCache",
    "build_cache",
    # Simulation
    "SimulationRunner",
    "SimulationConfig",
    "SimulationResult",
    # Content
    "ContentCatalog",
    "ContentItem",
    # Evaluation
    "EvalMetrics",
    "compute_metrics",
    "run_benchmark",
]
