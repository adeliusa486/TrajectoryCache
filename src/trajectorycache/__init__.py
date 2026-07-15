"""
Spatial-urgency-aware edge cache for vehicular networks.

The primary spatial policy is ``SpatialUrgencyCache`` (SU), formerly named
``TrajectoryCache`` (still importable as a backward-compatible alias).

Quick start
-----------
>>> from trajectorycache import SpatialUrgencyCache, SimulationRunner, SimulationConfig
>>> cache = SpatialUrgencyCache(capacity=20, urgency_weight=0.5)
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
    SpatialUrgencyCache,
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
    "SpatialUrgencyCache",
    "TrajectoryCache",  # deprecated alias
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
