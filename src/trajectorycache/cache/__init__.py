"""Cache replacement policy implementations."""

from .base import BaseCache, CacheItem
from .adaptive import AdaptiveTrajectoryCache
from .baselines import FIFOCache, LFUCache, ProximityCache, RandomCache
from .lru import LRUCache
from .trajectory import TrajectoryCache
from .learned import QLearningCache

__all__ = [
    "BaseCache",
    "CacheItem",
    "TrajectoryCache",
    "AdaptiveTrajectoryCache",
    "LRUCache",
    "LFUCache",
    "RandomCache",
    "FIFOCache",
    "ProximityCache",
    "QLearningCache",
]

REGISTRY: dict[str, type[BaseCache]] = {
    "trajectory": TrajectoryCache,
    "adaptive": AdaptiveTrajectoryCache,
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
