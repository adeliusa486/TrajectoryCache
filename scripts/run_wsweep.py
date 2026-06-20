#!/usr/bin/env python
"""
scripts/run_wsweep.py

Sweep the urgency weight W across {0.1, 0.2, 0.3, 0.5, 0.7, 0.9}
and report mean miss rates (averaged over multiple seeds) for both
TC and LFU. Produces Figure 3 data for the paper.

Usage
-----
    python scripts/run_wsweep.py
    python scripts/run_wsweep.py --seeds 42 7 13 --output experiments/results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trajectorycache.evaluation.benchmark import run_benchmark
from trajectorycache.simulation.runner import SimulationConfig
from trajectorycache.utils.config import load_config
from trajectorycache.utils.logging import setup_logging

W_VALUES = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
PAPER_SEEDS = [84810, 15592, 4278, 98196, 37048, 33098, 30256, 19289, 97530, 14434]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="W-sweep for TrajectoryCache")
    p.add_argument("--config", type=Path, default=Path("configs/simulation.yaml"))
    p.add_argument("--output", type=Path, default=Path("experiments/results/wsweep"))
    p.add_argument("--seeds", type=int, nargs="+", default=PAPER_SEEDS)
    p.add_argument("--w-values", type=float, nargs="+", default=W_VALUES)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level="WARNING")

    cfg = load_config(args.config)
    cfg.platoon_size = 10  # Enforce platooning condition to match run_multiseed

    print(f"\nW-sweep | alpha={cfg.zipf_alpha} | seeds={args.seeds}")
    print(f"W values: {args.w_values}\n")

    results_tc: dict[float, list[float]] = {w: [] for w in args.w_values}
    results_lfu: list[float] = []

    for seed in args.seeds:
        cfg.seed = seed
        # Run LFU once per seed (W-independent)
        lfu_result = run_benchmark(
            config=cfg,
            policies=[("lfu", {"pop_window": 300.0})],
            output_dir=None,
        )
        results_lfu.append(lfu_result["LFU"].miss_rate * 100.0)

        for w in args.w_values:
            res = run_benchmark(
                config=cfg,
                policies=[("trajectory", {"urgency_weight": w})],
                output_dir=None,
            )
            results_tc[w].append(res["TrajectoryCache"].miss_rate * 100.0)
            print(
                f"  seed={seed}  W={w:.1f}  TC={results_tc[w][-1]:.2f}%  "
                f"LFU={results_lfu[-1]:.2f}%"
            )

    # Summary table
    print(f"\n{'W':>6}  {'TC mean%':>10}  {'TC std%':>8}  {'LFU mean%':>10}")
    print("-" * 44)
    lfu_mean = float(np.mean(results_lfu))
    lfu_std = float(np.std(results_lfu))

    sweep_data = {"lfu_mean": round(lfu_mean, 4), "lfu_std": round(lfu_std, 4), "w_sweep": {}}
    for w in args.w_values:
        vals = results_tc[w]
        m = float(np.mean(vals))
        s = float(np.std(vals))
        print(f"  {w:.1f}  {m:>10.2f}%  {s:>7.2f}%  {lfu_mean:>10.2f}%")
        sweep_data["w_sweep"][str(w)] = {"mean": round(m, 4), "std": round(s, 4)}

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    fname = out / "wsweep_results.json"
    with open(fname, "w") as f:
        json.dump(sweep_data, f, indent=2)
    print(f"\nSaved to {fname}")


if __name__ == "__main__":
    main()
