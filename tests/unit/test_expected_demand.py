"""Unit tests for ExpectedDemandCache (EDC)."""

from __future__ import annotations

import pytest

from trajectorycache.cache import ExpectedDemandCache, build_cache


def test_registry_builds_edc():
    c = build_cache("expected_demand", capacity=10)
    assert isinstance(c, ExpectedDemandCache)


def test_hit_then_miss_accounting():
    c = ExpectedDemandCache(capacity=5)
    assert c.request(1, 100.0, 0.0, vehicles=[], catalog={1: 100.0}) is False  # miss (insert)
    assert c.request(1, 100.0, 1.0, vehicles=[], catalog={1: 100.0}) is True   # hit
    assert c.hits == 1 and c.misses == 1


def test_fills_to_capacity_without_eviction():
    c = ExpectedDemandCache(capacity=3)
    for i in range(3):
        c.request(i, float(i), 0.0, vehicles=[], catalog={})
    assert len(c) == 3
    for i in range(3):
        assert i in c


def test_capacity_never_exceeded():
    c = ExpectedDemandCache(capacity=4)
    for i in range(30):
        c.request(i, float(i * 10), float(i), vehicles=[], catalog={})
    assert len(c) <= 4


def test_exposure_only_counts_forward_window():
    c = ExpectedDemandCache(capacity=5, r_req=800.0)
    # vehicle at x=4500 heading +x; item at 5000 is 500 m ahead -> exposed
    ahead = [{"x": 4500.0, "speed": 30.0, "direction": 1}]
    assert c._exposure(5000.0, ahead, 0.0) == 1.0
    # item at 4000 is behind the vehicle -> not exposed
    assert c._exposure(4000.0, ahead, 0.0) == 0.0
    # item at 5400 is 900 m ahead, beyond r_req=800 -> not exposed
    assert c._exposure(5400.0, ahead, 0.0) == 0.0


def test_exposure_ignores_stopped_vehicles():
    c = ExpectedDemandCache(capacity=5)
    stopped = [{"x": 4500.0, "speed": 0.0, "direction": 1}]
    assert c._exposure(5000.0, stopped, 0.0) == 0.0


def test_degrades_to_popularity_when_no_vehicles():
    """With zero exposure everywhere, EDC must rank purely by windowed count,
    i.e. evict the least-frequently-requested item (LFU-equivalent)."""
    c = ExpectedDemandCache(capacity=2)
    cat = {1: 0.0, 2: 0.0, 3: 0.0}
    # item 1 requested 3x, item 2 requested 1x, both cached
    for t in range(3):
        c.request(1, 0.0, float(t), vehicles=[], catalog=cat)
    c.request(2, 0.0, 3.0, vehicles=[], catalog=cat)
    assert 1 in c and 2 in c and len(c) == 2
    # new item 3 (count becomes 1 on this request) vs cached victim:
    # item 2 has count 1, item 1 has count 3 -> victim is item 2, tie with item 3
    # (both count 1); eviction requires victim strictly < new, so item 2 (1) is
    # NOT < item 3 (1): new item is discarded, popular item 1 retained.
    c.request(3, 0.0, 4.0, vehicles=[], catalog=cat)
    assert 1 in c, "most popular item must never be evicted with no spatial signal"


def test_exposure_breaks_ties_toward_reachable_item():
    """Two equally-popular cached items; the one with no approaching vehicle
    should be evicted in favour of a new item that IS reachable."""
    c = ExpectedDemandCache(capacity=2, r_req=800.0)
    cat = {1: 1000.0, 2: 9000.0, 3: 5000.0}
    # seed equal popularity for items 1 and 2
    c.request(1, 1000.0, 0.0, vehicles=[], catalog=cat)
    c.request(2, 9000.0, 0.0, vehicles=[], catalog=cat)
    # now a vehicle approaches item 3 (at 5000) and item 1 (at 1000), not item 2
    veh = [{"x": 4800.0, "speed": 30.0, "direction": 1},   # approaches 5000
           {"x": 800.0, "speed": 30.0, "direction": 1}]    # approaches 1000
    # request item 3: it has count 1 and exposure 1 -> score 1*(1+1)=2;
    # item 2 has count 1, exposure 0 -> score 1; item 1 count 1 exposure 1 -> 2.
    # victim = item 2 (score 1) < new item 3 (score 2) -> evict item 2.
    c.request(3, 5000.0, 1.0, vehicles=veh, catalog=cat)
    assert 2 not in c, "unreachable item should be evicted for a reachable one"
    assert 3 in c


def test_invalid_damping_raises():
    with pytest.raises(ValueError):
        ExpectedDemandCache(capacity=5, damping="bogus")
