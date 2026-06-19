"""Unit tests for TrajectoryCache."""
from __future__ import annotations

import pytest

from trajectorycache.cache import TrajectoryCache
from trajectorycache.cache.base import CacheItem


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_cache() -> TrajectoryCache:
    return TrajectoryCache(capacity=3, urgency_weight=0.5)


VEHICLES = [
    {"x": 100.0, "speed": 20.0, "direction": 1},
    {"x": 800.0, "speed": 15.0, "direction": -1},
]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_invalid_capacity():
    with pytest.raises(ValueError):
        TrajectoryCache(capacity=0)


def test_invalid_urgency_weight_low():
    with pytest.raises(ValueError):
        TrajectoryCache(capacity=5, urgency_weight=-0.1)


def test_invalid_urgency_weight_high():
    with pytest.raises(ValueError):
        TrajectoryCache(capacity=5, urgency_weight=1.1)


def test_default_empty(small_cache):
    assert len(small_cache) == 0
    assert small_cache.hits == 0
    assert small_cache.misses == 0


# ---------------------------------------------------------------------------
# Basic hit / miss behaviour
# ---------------------------------------------------------------------------


def test_first_request_is_miss(small_cache):
    hit = small_cache.request(0, 500.0, 0.0)
    assert not hit
    assert small_cache.misses == 1
    assert small_cache.hits == 0


def test_second_request_is_hit(small_cache):
    small_cache.request(0, 500.0, 0.0)
    hit = small_cache.request(0, 500.0, 1.0)
    assert hit
    assert small_cache.hits == 1


def test_different_items_miss(small_cache):
    for i in range(3):
        hit = small_cache.request(i, float(i * 100), float(i))
        assert not hit


# ---------------------------------------------------------------------------
# Capacity & eviction
# ---------------------------------------------------------------------------


def test_capacity_not_exceeded(small_cache):
    for i in range(10):
        small_cache.request(i, float(i * 100), float(i), vehicles=VEHICLES)
    assert len(small_cache) <= small_cache.capacity


def test_eviction_reduces_size(small_cache):
    for i in range(3):
        small_cache.request(i, float(i * 100), float(i))
    assert len(small_cache) == 3
    evicted = small_cache.evict()
    assert evicted is not None
    assert len(small_cache) == 2


def test_force_evict_empty_returns_none():
    cache = TrajectoryCache(capacity=5)
    assert cache.evict() is None


# ---------------------------------------------------------------------------
# Hit / miss rate arithmetic
# ---------------------------------------------------------------------------


def test_hit_rate_calculation(small_cache):
    small_cache.request(0, 0.0, 0.0)   # miss
    small_cache.request(0, 0.0, 1.0)   # hit
    small_cache.request(0, 0.0, 2.0)   # hit
    assert small_cache.hit_rate == pytest.approx(2 / 3)
    assert small_cache.miss_rate == pytest.approx(1 / 3)


def test_reset_stats(small_cache):
    small_cache.request(0, 0.0, 0.0)
    small_cache.reset_stats()
    assert small_cache.hits == 0
    assert small_cache.misses == 0
    assert small_cache.total_requests == 0


# ---------------------------------------------------------------------------
# Popularity window
# ---------------------------------------------------------------------------


def test_popularity_prune():
    cache = TrajectoryCache(capacity=10, pop_window=10.0)
    cache.request(0, 100.0, 0.0)   # t=0 - inside window at t=5
    cache.request(0, 100.0, 5.0)   # t=5 - inside window at t=15? no, 15-10=5  0
    cache.request(0, 100.0, 20.0)  # prune: t=20, cutoff=10 -> first two pruned
    counts = cache.popularity_counts()
    # Only the t=20 entry survives
    assert counts.get(0, 0) == 1


# ---------------------------------------------------------------------------
# Urgency computation
# ---------------------------------------------------------------------------


def test_raw_urgency_nearby_vehicle():
    cache = TrajectoryCache(capacity=10, t_pred=5.0, r_rel=200.0, alpha_d=0.5)
    # Vehicle at x=300, speed=20, direction=+1 -> predicted x=400 -> within 200m of 500
    vehicles = [{"x": 300.0, "speed": 20.0, "direction": 1}]
    urgency = cache._raw_urgency(500.0, vehicles, t=0.0)
    assert urgency > 0.0


def test_raw_urgency_far_vehicle():
    cache = TrajectoryCache(capacity=10, t_pred=1.0, r_rel=50.0)
    vehicles = [{"x": 0.0, "speed": 5.0, "direction": 1}]
    # predicted x = 5 -> |5 - 5000| >> 50  -> no contribution
    urgency = cache._raw_urgency(5000.0, vehicles, t=0.0)
    assert urgency == pytest.approx(0.0)


def test_raw_urgency_zero_speed_skipped():
    cache = TrajectoryCache(capacity=10)
    vehicles = [{"x": 100.0, "speed": 0.0, "direction": 1}]
    urgency = cache._raw_urgency(100.0, vehicles, t=0.0)
    assert urgency == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Score introspection
# ---------------------------------------------------------------------------


def test_get_scores_returns_dict(small_cache):
    for i in range(3):
        small_cache.request(i, float(i * 200), float(i))
    scores = small_cache.get_scores(VEHICLES, t=10.0)
    assert len(scores) == 3
    for v in scores.values():
        assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# Summary dict
# ---------------------------------------------------------------------------


def test_summary_keys(small_cache):
    summary = small_cache.summary()
    required = {"policy", "capacity", "size", "hits", "misses", "hit_rate", "miss_rate"}
    assert required.issubset(summary.keys())


def test_clear_resets_everything(small_cache):
    small_cache.request(0, 0.0, 0.0)
    small_cache.clear()
    assert len(small_cache) == 0
    assert small_cache.hits == 0
    assert small_cache.misses == 0
