"""Cache replacement policy implementations."""

from .base import BaseCache, CacheItem
from .adaptive import AdaptiveSpatialUrgencyCache, AdaptiveTrajectoryCache
from .baselines import FIFOCache, LFUCache, ProximityCache, RandomCache
from .expected_demand import ExpectedDemandCache
from .lru import LRUCache
from .trajectory import SpatialUrgencyCache, TrajectoryCache
from .learned import QLearningCache

__all__ = [
    "BaseCache",
    "CacheItem",
    "SpatialUrgencyCache",
    "TrajectoryCache",  # deprecated alias of SpatialUrgencyCache
    "AdaptiveSpatialUrgencyCache",
    "AdaptiveTrajectoryCache",  # deprecated alias
    "ExpectedDemandCache",
    "LRUCache",
    "LFUCache",
    "RandomCache",
    "FIFOCache",
    "ProximityCache",
    "QLearningCache",
]

REGISTRY: dict[str, type[BaseCache]] = {
    "su": SpatialUrgencyCache,
    "trajectory": SpatialUrgencyCache,  # deprecated alias key
    "adaptive": AdaptiveSpatialUrgencyCache,
    "expected_demand": ExpectedDemandCache,
    "lru": LRUCache,
    "lfu": LFUCache,
    "random": RandomCache,
    "fifo": FIFOCache,
    "proximity": ProximityCache,
    "qlearning": QLearningCache,
}


def build_cache(policy: str, capacity: int, **kwargs) -> BaseCache:
    """Factory: build a cache by policy name."""
    key = policy.lower()
    if key not in REGISTRY:
        raise ValueError(f"Unknown cache policy '{policy}'. Choose from: {list(REGISTRY)}")
    return REGISTRY[key](capacity=capacity, **kwargs)
