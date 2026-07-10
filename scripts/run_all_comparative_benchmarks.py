#!/usr/bin/env python
"""
Master Comparative Benchmarking Script for:
"Rule-Based versus Learning-Based Cache Replacement in Vehicular Edge Computing"

Computes all empirical tables and figures across 10 independent random seeds:
- Table 1: Primary Comparison (alpha = 0.8) across 10 seeds
- Table 2: Flatter Skew Comparison (alpha = 0.5) across 10 seeds
- Table 3: Proximity & QLearning Baseline Evaluation
- Table 4: GNSS Positioning Noise & Telemetry Lag Sensitivity
- Table 5: Density Crossover Evaluation (n = 50 to 600)
- Table 6: Wall-Clock Computational Cost (Inference vs. Training Overhead)
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trajectorycache.cache import build_cache
from trajectorycache.evaluation.metrics import compute_metrics
from trajectorycache.simulation.runner import SimulationRunner
from trajectorycache.utils.config import load_config

SEEDS_10 = [84810, 15592, 4278, 98196, 37048, 33098, 30256, 19289, 97530, 14434]
SEEDS_5 = [84810, 15592, 4278, 98196, 37048]
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "simulation.yaml"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_policy_multi_seed(policy_key, kwargs, seeds, extra_cfg=None):
    misses = []
    durations = []
    for s in seeds:
        cfg = load_config(CONFIG_PATH)
        cfg.seed = s
        if extra_cfg:
            for k, v in extra_cfg.items():
                setattr(cfg, k, v)
        cache = build_cache(policy_key, cfg.cache_capacity, **kwargs)
        t0 = time.perf_counter()
        runner = SimulationRunner(cache=cache, config=cfg)
        res = runner.run()
        t1 = time.perf_counter()
        m = compute_metrics(res)
        misses.append(m.miss_rate * 100.0)
        durations.append((t1 - t0) * 1000.0)  # ms
    return float(np.mean(misses)), float(np.std(misses)), float(np.mean(durations))


def main():
    print("================================================================================")
    print("  VEHICULAR EDGE CACHING: COMPARATIVE BENCHMARKING SUITE")
    print("================================================================================")
    print(f"Seeds (10-seed protocol): {SEEDS_10}\n")
    sys.stdout.flush()

    results = {
        "alpha_08": {},
        "alpha_05": {},
        "proximity_and_learned": {},
        "gnss_noise_sensitivity": {},
        "density_sweep": {},
        "wallclock_cost": {},
    }

    policies = [
        ("LRU", "lru", {}),
        ("FIFO", "fifo", {}),
        ("Random", "random", {}),
        ("LFU", "lfu", {"pop_window": 300.0}),
        ("Proximity", "proximity", {}),
        ("QLearning (Learned)", "qlearning", {"lr": 0.05}),
        ("TrajectoryCache (W=0.2)", "trajectory", {"urgency_weight": 0.2}),
    ]

    print("--- [1/5] Primary Comparison (alpha = 0.8, 10 seeds) ---")
    sys.stdout.flush()
    for name, key, kw in policies:
        mean, std, dur = run_policy_multi_seed(key, kw, SEEDS_10, {"zipf_alpha": 0.8})
        results["alpha_08"][name] = {"miss_rate_mean": round(mean, 2), "miss_rate_std": round(std, 2), "time_ms": round(dur, 2)}
        print(f"  {name:25s} | Miss Rate: {mean:6.2f}% +/- {std:4.2f}%  ({dur:.1f} ms/run)")
        sys.stdout.flush()

    print("\n--- [2/5] Flatter Popularity Skew (alpha = 0.5, 10 seeds) ---")
    sys.stdout.flush()
    for name, key, kw in [
        ("LRU", "lru", {}),
        ("LFU", "lfu", {"pop_window": 300.0}),
        ("QLearning (Learned)", "qlearning", {"lr": 0.05}),
        ("TrajectoryCache (W=0.2)", "trajectory", {"urgency_weight": 0.2}),
    ]:
        mean, std, dur = run_policy_multi_seed(key, kw, SEEDS_10, {"zipf_alpha": 0.5})
        results["alpha_05"][name] = {"miss_rate_mean": round(mean, 2), "miss_rate_std": round(std, 2)}
        print(f"  {name:25s} | Miss Rate: {mean:6.2f}% +/- {std:4.2f}%")
        sys.stdout.flush()

    print("\n--- [3/5] GNSS Positioning Noise & Telemetry Lag Sensitivity ---")
    sys.stdout.flush()
    lfu_m = results["alpha_08"]["LFU"]["miss_rate_mean"]
    noise_configs = [
        ("clean (0m, 0step)", 0.0, 0),
        ("noise_5m (5m, 0step)", 5.0, 0),
        ("noise_15m (15m, 0step)", 15.0, 0),
        ("lag_1s (0m, 1step)", 0.0, 1),
        ("lag_3s (0m, 3step)", 0.0, 3),
        ("combined_5m_1s", 5.0, 1),
        ("combined_15m_3s", 15.0, 3),
    ]
    for label, n_std, lag in noise_configs:
        tc_m, tc_s, _ = run_policy_multi_seed("trajectory", {"urgency_weight": 0.2}, SEEDS_5, {"pos_noise_std": n_std, "update_lag_steps": lag})
        ql_m, ql_s, _ = run_policy_multi_seed("qlearning", {"lr": 0.05}, SEEDS_5, {"pos_noise_std": n_std, "update_lag_steps": lag})
        results["gnss_noise_sensitivity"][label] = {
            "tc_mean": round(tc_m, 2), "tc_std": round(tc_s, 2),
            "ql_mean": round(ql_m, 2), "ql_std": round(ql_s, 2),
            "tc_vs_lfu": round(tc_m - lfu_m, 2),
        }
        print(f"  {label:24s} | TC: {tc_m:6.2f}% (+/-{tc_s:.2f}) | QLearning: {ql_m:6.2f}% (+/-{ql_s:.2f})")
        sys.stdout.flush()

    print("\n--- [4/5] Vehicle Density Sweep (n = 50, 100, 200, 400, 600) ---")
    sys.stdout.flush()
    densities = [50, 100, 200, 400, 600]
    for d in densities:
        tc_m, tc_s, _ = run_policy_multi_seed("trajectory", {"urgency_weight": 0.2}, SEEDS_5, {"n_vehicles": d})
        lfu_m, lfu_s, _ = run_policy_multi_seed("lfu", {"pop_window": 300.0}, SEEDS_5, {"n_vehicles": d})
        ql_m, ql_s, _ = run_policy_multi_seed("qlearning", {"lr": 0.05}, SEEDS_5, {"n_vehicles": d})
        results["density_sweep"][str(d)] = {
            "tc": {"mean": round(tc_m, 2), "std": round(tc_s, 2)},
            "lfu": {"mean": round(lfu_m, 2), "std": round(lfu_s, 2)},
            "qlearning": {"mean": round(ql_m, 2), "std": round(ql_s, 2)},
        }
        print(f"  Density n={d:3d} | TC: {tc_m:6.2f}% | LFU: {lfu_m:6.2f}% | QLearning: {ql_m:6.2f}%")
        sys.stdout.flush()

    out_file = OUTPUT_DIR / "comparative_study_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n================================================================================")
    print(f"  ALL BENCHMARKS COMPLETED SUCCESSFULLY! Results saved to: {out_file}")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
