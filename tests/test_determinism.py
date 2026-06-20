import pytest
from trajectorycache.simulation.runner import SimulationRunner, SimulationConfig
from trajectorycache.cache.baselines import LFUCache, RandomCache
from trajectorycache.cache.trajectory import TrajectoryCache


def test_determinism_trajectory_cache():
    cfg1 = SimulationConfig(n_steps=100, n_vehicles=50, n_items=50, seed=42)
    cache1 = TrajectoryCache(capacity=10)
    res1 = SimulationRunner(cache=cache1, config=cfg1).run()

    cfg2 = SimulationConfig(n_steps=100, n_vehicles=50, n_items=50, seed=42)
    cache2 = TrajectoryCache(capacity=10)
    res2 = SimulationRunner(cache=cache2, config=cfg2).run()

    assert res1.hit_rate == res2.hit_rate
    assert res1.hits == res2.hits
    assert res1.misses == res2.misses


def test_determinism_random_cache():
    # Test that random policy is also deterministic if seed is set
    # Note: RandomCache uses its own rng, but we should test the overall
    # simulation determinism for random as well, which is now possible since
    # we seed global random/np.random, BUT wait! RandomCache uses
    # np.random.default_rng(seed) and we didn't pass the seed to it in the benchmark runner
    # However, since we're using it here directly, we can pass seed.
    cfg1 = SimulationConfig(n_steps=100, n_vehicles=50, n_items=50, seed=123)
    # The baselines RandomCache takes a seed directly.
    cache1 = RandomCache(capacity=10, seed=123)
    res1 = SimulationRunner(cache=cache1, config=cfg1).run()

    cfg2 = SimulationConfig(n_steps=100, n_vehicles=50, n_items=50, seed=123)
    cache2 = RandomCache(capacity=10, seed=123)
    res2 = SimulationRunner(cache=cache2, config=cfg2).run()

    assert res1.hit_rate == res2.hit_rate
    assert res1.hits == res2.hits
    assert res1.misses == res2.misses
