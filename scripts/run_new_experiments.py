#!/usr/bin/env python
"""
New experiments for the Ad Hoc Networks revision:
  1. Proximity baseline (mobility-aware, no-popularity heuristic) added
     alongside LRU/FIFO/Random/LFU/TC.
  2. GNSS/V2X positioning-error + update-latency sensitivity for TC.

Uses configs/simulation.yaml (the config verified to reproduce the paper's
Table 1 / Table 2 numbers exactly) plus the first 5 of the paper's 10 seeds,
matching the convention used for the ablation/density studies (Sec. 5.3).
"""
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from trajectorycache.cache import build_cache
from trajectorycache.utils.config import load_config
from trajectorycache.simulation.runner import SimulationRunner
from trajectorycache.evaluation.metrics import compute_metrics

SEEDS = [84810, 15592, 4278, 98196, 37048]
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "simulation.yaml"


def run_multi_seed(policy, kwargs, extra_cfg=None):
    misses = []
    for s in SEEDS:
        cfg = load_config(CONFIG_PATH)
        cfg.seed = s
        if extra_cfg:
            for k, v in extra_cfg.items():
                setattr(cfg, k, v)
        cache = build_cache(policy, cfg.cache_capacity, **kwargs)
        m = compute_metrics(SimulationRunner(cache=cache, config=cfg).run())
        misses.append(m.miss_rate * 100)
    return float(np.mean(misses)), float(np.std(misses))


def main():
    t0 = time.time()
    out = {"experiment_1_baseline_comparison": {}, "experiment_2_gnss_sensitivity": {}}

    print("=== Experiment 1: full baseline comparison (incl. Proximity) ===")
    policies = [
        ("LRU", "lru", {}),
        ("FIFO", "fifo", {}),
        ("Random", "random", {}),
        ("LFU", "lfu", {}),
        ("Proximity", "proximity", {}),
        ("QLearning", "qlearning", {"lr": 0.05}),
        ("TC (W=0.2)", "trajectory", {"urgency_weight": 0.2}),
    ]
    for label, key, kw in policies:
        mean, std = run_multi_seed(key, kw)
        out["experiment_1_baseline_comparison"][label] = {"mean": mean, "std": std}
        print(f"  {label:15s} {mean:6.2f} +/- {std:.2f}")

    print("\n=== Experiment 2: GNSS/V2X noise + update-lag sensitivity (TC vs LFU vs QLearning) ===")
    lfu_mean, lfu_std = run_multi_seed("lfu", {})
    out["experiment_2_gnss_sensitivity"]["LFU_reference"] = {"mean": lfu_mean, "std": lfu_std}
    print(f"  LFU reference (telemetry-immune): {lfu_mean:.2f} +/- {lfu_std:.2f}")

    configs = [
        ("clean", 0.0, 0),
        ("noise_2m", 2.0, 0),
        ("noise_5m", 5.0, 0),
        ("noise_10m", 10.0, 0),
        ("noise_15m", 15.0, 0),
        ("lag_1step", 0.0, 1),
        ("lag_3step", 0.0, 3),
        ("noise_5m_lag_1", 5.0, 1),
        ("noise_15m_lag_3", 15.0, 3),
    ]
    for label, noise, lag in configs:
        tc_mean, tc_std = run_multi_seed(
            "trajectory", {"urgency_weight": 0.2},
            extra_cfg={"pos_noise_std": noise, "update_lag_steps": lag},
        )
        ql_mean, ql_std = run_multi_seed(
            "qlearning", {"lr": 0.05},
            extra_cfg={"pos_noise_std": noise, "update_lag_steps": lag},
        )
        margin_tc = tc_mean - lfu_mean
        margin_ql = ql_mean - lfu_mean
        out["experiment_2_gnss_sensitivity"][label] = {
            "tc_mean": tc_mean, "tc_std": tc_std,
            "ql_mean": ql_mean, "ql_std": ql_std,
            "noise_std_m": noise, "lag_steps": lag,
            "margin_tc_vs_lfu_pp": margin_tc,
            "margin_ql_vs_lfu_pp": margin_ql,
        }
        print(f"  {label:18s} noise={noise:5.1f}m lag={lag}step | TC: {tc_mean:6.2f}+/-{tc_std:.2f} | QL: {ql_mean:6.2f}+/-{ql_std:.2f}")

    out["_meta"] = {
        "seeds": SEEDS, "elapsed_s": time.time() - t0,
        "config_path": str(CONFIG_PATH),
    }
    out_path = Path(__file__).resolve().parents[1] / "experiments" / "results" / "new_experiments.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {out_path}")
    print(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
