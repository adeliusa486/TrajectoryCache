#!/usr/bin/env python
"""
scripts/sweep.py

Grid-search over TrajectoryCache hyperparameters (W, alpha_d, r_rel)
and report the best configuration by hit rate.

Usage
-----
    python scripts/sweep.py
    python scripts/sweep.py --config configs/sweep.yaml --output experiments/results/sweep
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trajectorycache import TrajectoryCache
from trajectorycache.evaluation.metrics import compute_metrics
from trajectorycache.simulation.runner import SimulationConfig, SimulationRunner
from trajectorycache.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TrajectoryCache hyperparameter sweep")
    p.add_argument("--config", type=Path, default=Path("configs/sweep.yaml"))
    p.add_argument("--output", type=Path, default=Path("experiments/results/sweep"))
    p.add_argument("--log-level", default="WARNING")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level)
    args.output.mkdir(parents=True, exist_ok=True)

    with open(args.config) as fh:
        cfg_raw = yaml.safe_load(fh)

    base = cfg_raw["base"]
    sweep = cfg_raw["sweep"]
    sim_raw = cfg_raw.get("simulation", {})

    sim_cfg = SimulationConfig(
        n_steps=sim_raw.get("n_steps", 500),
        warmup_steps=sim_raw.get("warmup_steps", 50),
        n_vehicles=sim_raw.get("n_vehicles", 50),
        n_items=sim_raw.get("n_items", 200),
        cache_capacity=sim_raw.get("cache_capacity", 20),
        seed=sim_raw.get("seed", 42),
    )

    W_values = sweep["urgency_weight"]
    alpha_vals = sweep["alpha_d"]
    r_rel_vals = sweep["r_rel"]

    combinations = list(itertools.product(W_values, alpha_vals, r_rel_vals))
    print(f"Sweeping {len(combinations)} combinations ")

    results = []
    best_hit_rate = -1.0
    best_params = {}

    for idx, (W, alpha_d, r_rel) in enumerate(combinations):
        cache = TrajectoryCache(
            capacity=sim_cfg.cache_capacity,
            urgency_weight=W,
            pop_window=base.get("pop_window", 300.0),
            t_pred=base.get("t_pred", 3.0),
            alpha_d=alpha_d,
            r_rel=r_rel,
        )
        runner = SimulationRunner(cache=cache, config=sim_cfg)
        result = runner.run(verbose=False)
        metrics = compute_metrics(result)

        entry = {
            "urgency_weight": W,
            "alpha_d": alpha_d,
            "r_rel": r_rel,
            "hit_rate": metrics.hit_rate,
            "miss_rate": metrics.miss_rate,
        }
        results.append(entry)

        if metrics.hit_rate > best_hit_rate:
            best_hit_rate = metrics.hit_rate
            best_params = entry

        if (idx + 1) % 20 == 0 or (idx + 1) == len(combinations):
            print(f"  [{idx+1}/{len(combinations)}] best so far: {best_params}")

    # Save full sweep results
    out_path = args.output / "sweep_results.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"\n{'='*55}")
    print(f"Best configuration (hit_rate={best_hit_rate*100:.2f}%):")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"\nFull results saved to: {out_path}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
