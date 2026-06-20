"""Unit tests for LRU, LFU, Random, and FIFO baselines."""

from __future__ import annotations

import pytest

from trajectorycache.cache import FIFOCache, LFUCache, LRUCache, RandomCache, build_cache

# ---------------------------------------------------------------------------
# LRU
# ---------------------------------------------------------------------------


def test_lru_hit():
    c = LRUCache(capacity=3)
    c.request(1, 0.0, 0.0)
    assert c.request(1, 0.0, 1.0) is True


def test_lru_evicts_least_recent():
    c = LRUCache(capacity=2)
    c.request(1, 0.0, 0.0)  # insert 1
    c.request(2, 0.0, 1.0)  # insert 2
    c.request(1, 0.0, 2.0)  # access 1 -> 1 is MRU, 2 is LRU
    c.request(3, 0.0, 3.0)  # insert 3 -> should evict 2
    assert 2 not in c
    assert 1 in c
    assert 3 in c


def test_lru_capacity_not_exceeded():
    c = LRUCache(capacity=5)
    for i in range(20):
        c.request(i, 0.0, float(i))
    assert len(c) <= 5


# ---------------------------------------------------------------------------
# LFU
# ---------------------------------------------------------------------------


def test_lfu_evicts_least_frequent():
    c = LFUCache(capacity=2)
    c.request(1, 0.0, 0.0)  # freq[1]=1
    c.request(2, 0.0, 1.0)  # freq[2]=1
    c.request(1, 0.0, 2.0)  # freq[1]=2
    c.request(3, 0.0, 3.0)  # evict item with lowest freq -> item 2 (freq=1)
    assert 2 not in c
    assert 1 in c
    assert 3 in c


def test_lfu_stats():
    c = LFUCache(capacity=5)
    for _ in range(10):
        c.request(1, 0.0, 0.0)  # 1 miss + 9 hits
    assert c.hits == 9
    assert c.misses == 1


# ---------------------------------------------------------------------------
# Random
# ---------------------------------------------------------------------------


def test_random_capacity():
    c = RandomCache(capacity=3)
    for i in range(50):
        c.request(i, 0.0, float(i))
    assert len(c) <= 3


def test_random_hit():
    c = RandomCache(capacity=10)
    c.request(99, 0.0, 0.0)
    hit = c.request(99, 0.0, 1.0)
    assert hit is True


# ---------------------------------------------------------------------------
# FIFO
# ---------------------------------------------------------------------------


def test_fifo_evicts_first_inserted():
    c = FIFOCache(capacity=2)
    c.request(1, 0.0, 0.0)  # first in
    c.request(2, 0.0, 1.0)
    c.request(3, 0.0, 2.0)  # should evict item 1
    assert 1 not in c
    assert 2 in c
    assert 3 in c


def test_fifo_hit_does_not_reset_order():
    c = FIFOCache(capacity=2)
    c.request(1, 0.0, 0.0)
    c.request(2, 0.0, 1.0)
    c.request(1, 0.0, 2.0)  # hit - should NOT move 1 to back
    c.request(3, 0.0, 3.0)  # evict: item 1 is still first-in
    assert 1 not in c


# ---------------------------------------------------------------------------
# Registry / factory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("policy", ["lru", "lfu", "random", "fifo", "trajectory"])
def test_build_cache_factory(policy):
    cache = build_cache(policy, capacity=10)
    assert cache.capacity == 10


def test_build_cache_unknown_raises():
    with pytest.raises(ValueError, match="Unknown cache policy"):
        build_cache("nonexistent", capacity=10)


# ---------------------------------------------------------------------------
# BaseCache shared properties
# ---------------------------------------------------------------------------


def test_contains_operator():
    c = LRUCache(capacity=5)
    c.request(7, 0.0, 0.0)
    assert 7 in c
    assert 99 not in c


def test_clear_empties_cache():
    c = LFUCache(capacity=5)
    for i in range(5):
        c.request(i, 0.0, float(i))
    c.clear()
    assert len(c) == 0
    assert c.hits == 0
    assert c.misses == 0


def test_miss_rate_empty_cache():
    c = LRUCache(capacity=5)
    assert c.miss_rate == 0.0
    assert c.hit_rate == 0.0
