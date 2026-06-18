"""Cache replacement policy implementations."""
from .base import BaseCache, CacheItem
from .baselines import FIFOCache, LFUCache, RandomCache
from .lru import LRUCache
from .trajectory import TrajectoryCache

__all__ = [
    "BaseCache",
    "CacheItem",
    "TrajectoryCache",
    "LRUCache",
    "LFUCache",
    "RandomCache",
    "FIFOCache",
]

REGISTRY: dict[str, type[BaseCache]] = {
    "trajectory": TrajectoryCache,
    "lru": LRUCache,
    "lfu": LFUCache,
    "random": RandomCache,
    "fifo": FIFOCache,
}


def build_cache(policy: str, capacity: int, **kwargs) -> BaseCache:
    """Factory: build a cache by policy name."""
    key = policy.lower()
    if key not in REGISTRY:
        raise ValueError(f"Unknown cache policy '{policy}'. Choose from: {list(REGISTRY)}")
    return REGISTRY[key](capacity=capacity, **kwargs)
