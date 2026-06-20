"""Integration tests: end-to-end simulation and benchmark runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trajectorycache import TrajectoryCache, build_cache, run_benchmark
from trajectorycache.evaluation.metrics import compute_metrics, save_results
from trajectorycache.simulation.runner import SimulationConfig, SimulationRunner

# ---------------------------------------------------------------------------
# Single-policy simulation
# ---------------------------------------------------------------------------


@pytest.fixture
def fast_config() -> SimulationConfig:
    """Minimal config for fast test runs."""
    return SimulationConfig(
        n_steps=100,
        warmup_steps=10,
        n_vehicles=10,
        n_items=50,
        cache_capacity=10,
        seed=42,
    )


@pytest.mark.parametrize("policy", ["trajectory", "lru", "lfu", "random", "fifo"])
def test_policy_simulation_completes(policy, fast_config):
    cache = build_cache(policy, capacity=fast_config.cache_capacity)
    runner = SimulationRunner(cache=cache, config=fast_config)
    result = runner.run()

    assert result.total_requests > 0
    assert result.hits + result.misses == result.total_requests
    assert 0.0 <= result.hit_rate <= 1.0
    assert 0.0 <= result.miss_rate <= 1.0
    assert result.hit_rate + result.miss_rate == pytest.approx(1.0)


def test_trajectory_hit_rate_nonnegative(fast_config):
    cache = TrajectoryCache(capacity=10, urgency_weight=0.5)
    runner = SimulationRunner(cache=cache, config=fast_config)
    result = runner.run()
    assert result.hit_rate >= 0.0


def test_per_step_hit_rate_length(fast_config):
    cache = build_cache("lru", capacity=10)
    runner = SimulationRunner(cache=cache, config=fast_config)
    result = runner.run()
    assert len(result.per_step_hit_rate) == fast_config.n_steps


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def test_benchmark_returns_all_policies(fast_config):
    results = run_benchmark(config=fast_config, verbose=False)
    for name in ["TrajectoryCache", "LRU", "LFU", "Random", "FIFO"]:
        assert name in results


def test_benchmark_metrics_valid(fast_config):
    results = run_benchmark(config=fast_config)
    for name, metrics in results.items():
        assert 0.0 <= metrics.hit_rate <= 1.0
        assert metrics.total_requests > 0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_save_and_load_results(tmp_path, fast_config):
    from trajectorycache.evaluation.metrics import load_results

    results = run_benchmark(config=fast_config)
    metrics_list = list(results.values())
    out = tmp_path / "results.json"
    save_results(metrics_list, out)

    assert out.exists()
    loaded = load_results(out)
    assert len(loaded) == len(metrics_list)
    assert loaded[0].policy == metrics_list[0].policy


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def test_config_load_defaults(tmp_path):
    from trajectorycache.utils.config import load_config

    cfg = load_config(path=tmp_path / "nonexistent.yaml")
    assert cfg.n_steps == SimulationConfig().n_steps


def test_config_round_trip(tmp_path):
    from trajectorycache.utils.config import load_config, save_config

    cfg = SimulationConfig(n_steps=777, seed=99)
    path = tmp_path / "test_config.yaml"
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded.n_steps == 777
    assert loaded.seed == 99
