"""Unit tests for highway simulation and content catalog."""
from __future__ import annotations

import pytest

from trajectorycache.content.catalog import ContentCatalog
from trajectorycache.simulation.highway import HighwaySimulation, Vehicle


# ---------------------------------------------------------------------------
# Vehicle
# ---------------------------------------------------------------------------


def test_vehicle_forward_step():
    v = Vehicle(vehicle_id=0, x=100.0, speed=20.0, direction=1)
    v.step(1.0)
    assert v.x == pytest.approx(120.0)


def test_vehicle_backward_step():
    v = Vehicle(vehicle_id=0, x=100.0, speed=20.0, direction=-1)
    v.step(1.0)
    assert v.x == pytest.approx(80.0)


def test_vehicle_to_dict_keys():
    v = Vehicle(vehicle_id=5, x=50.0, speed=10.0, direction=1)
    d = v.to_dict()
    assert set(d.keys()) == {"id", "x", "speed", "direction", "lane"}


# ---------------------------------------------------------------------------
# HighwaySimulation
# ---------------------------------------------------------------------------


def test_highway_initial_vehicle_count():
    sim = HighwaySimulation(n_vehicles=30, seed=0)
    assert len(sim.vehicles) == 30


def test_highway_step_advances_time():
    sim = HighwaySimulation(dt=1.0, seed=0)
    sim.step()
    assert sim.current_time == pytest.approx(1.0)
    sim.step()
    assert sim.current_time == pytest.approx(2.0)


def test_highway_step_returns_vehicle_dicts():
    sim = HighwaySimulation(n_vehicles=10, seed=0)
    states = sim.step()
    assert len(states) == 10
    for s in states:
        assert "x" in s and "speed" in s and "direction" in s


def test_highway_vehicles_stay_on_road():
    """Vehicles wrap around and stay within [0, road_length]."""
    sim = HighwaySimulation(road_length=1000.0, n_vehicles=20, seed=1)
    for _ in range(200):
        sim.step()
    for v in sim.vehicles:
        assert 0.0 <= v.x <= 1000.0


def test_highway_run_yields_correct_count():
    sim = HighwaySimulation(n_vehicles=5, seed=0)
    steps = list(sim.run(n_steps=10))
    assert len(steps) == 10
    for t, states in steps:
        assert t > 0
        assert len(states) == 5


def test_highway_snapshot_does_not_advance_time():
    sim = HighwaySimulation(seed=0)
    t_before = sim.current_time
    sim.snapshot()
    assert sim.current_time == t_before


# ---------------------------------------------------------------------------
# ContentCatalog
# ---------------------------------------------------------------------------


def test_catalog_size():
    cat = ContentCatalog(n_items=100, seed=0)
    assert len(cat) == 100


def test_catalog_locations_in_range():
    road_length = 5000.0
    cat = ContentCatalog(n_items=50, road_length=road_length, seed=0)
    for item in cat:
        assert 0.0 <= item.location <= road_length


def test_catalog_sample_request():
    cat = ContentCatalog(n_items=50, seed=0)
    item = cat.generate_requests(1)[0]  # sample_request() alias → generate_requests(1)[0]
    assert 0 <= item.item_id < 50



def test_catalog_generate_requests_count():
    cat = ContentCatalog(n_items=100, seed=0)
    reqs = cat.generate_requests(25)
    assert len(reqs) == 25


def test_catalog_location_map_keys():
    cat = ContentCatalog(n_items=10, seed=0)
    loc_map = cat.location_map()
    assert set(loc_map.keys()) == set(range(10))


def test_catalog_zipf_skewed():
    """Item 0 (most popular) should appear more than item 99."""
    cat = ContentCatalog(n_items=100, zipf_alpha=2.0, seed=7)
    reqs = cat.generate_requests(10_000)
    from collections import Counter
    counts = Counter(r.item_id for r in reqs)
    assert counts[0] > counts[99]
