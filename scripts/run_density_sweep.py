#!/usr/bin/env python
"""
scripts/run_density_sweep.py

Sweep vehicle density across {50, 100, 200, 400, 600} and report
TC vs LFU miss rates averaged over 5 seeds. Produces density analysis data.
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

DENSITIES = [50, 100, 200, 400, 600]
PAPER_SEEDS = [84810, 15592, 4278, 98196, 37048, 33098, 30256, 19289, 97530, 14434]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/simulation.yaml"))
    p.add_argument("--output", type=Path, default=Path("experiments/results/density"))
    # Paper Table 6 (density sweep) uses the first 5 of the 10 seeds.
    p.add_argument("--seeds", type=int, nargs="+", default=PAPER_SEEDS[:5])
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(level="WARNING")
    cfg = load_config(args.config)

    print(f"\nDensity sweep | alpha={cfg.zipf_alpha} | seeds={args.seeds}")
    print(f"Densities: {DENSITIES}\n")

    tc_means, lfu_means, ql_means, tc_stds, lfu_stds, ql_stds = [], [], [], [], [], []

    for n_veh in DENSITIES:
        tc_vals, lfu_vals, ql_vals = [], [], []
        cfg.n_vehicles = n_veh
        for seed in args.seeds:
            cfg.seed = seed
            res = run_benchmark(
                config=cfg,
                policies=[
                    ("trajectory", {"urgency_weight": 0.2}),
                    ("qlearning", {"lr": 0.05}),
                    ("lfu", {"pop_window": 300.0}),
                ],
                output_dir=None,
            )
            tc_vals.append(res["TrajectoryCache"].miss_rate * 100.0)
            lfu_vals.append(res["LFU"].miss_rate * 100.0)
            ql_vals.append(res["QLearning"].miss_rate * 100.0)

        tc_means.append(float(np.mean(tc_vals)))
        lfu_means.append(float(np.mean(lfu_vals)))
        ql_means.append(float(np.mean(ql_vals)))
        tc_stds.append(float(np.std(tc_vals)))
        lfu_stds.append(float(np.std(lfu_vals)))
        ql_stds.append(float(np.std(ql_vals)))
        print(f"Density={n_veh:3d} | TC: {tc_means[-1]:.2f}% +/- {tc_stds[-1]:.2f}% | LFU: {lfu_means[-1]:.2f}% +/- {lfu_stds[-1]:.2f}% | QLearning: {ql_means[-1]:.2f}% +/- {ql_stds[-1]:.2f}%")

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    data = {
        "densities": DENSITIES,
        "tc_means": [round(v, 4) for v in tc_means],
        "tc_stds": [round(v, 4) for v in tc_stds],
        "lfu_means": [round(v, 4) for v in lfu_means],
        "lfu_stds": [round(v, 4) for v in lfu_stds],
        "ql_means": [round(v, 4) for v in ql_means],
        "ql_stds": [round(v, 4) for v in ql_stds],
    }
    fname = out / "density_sweep.json"
    with open(fname, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved to {fname}")


if __name__ == "__main__":
    main()
