#!/usr/bin/env python
"""
scripts/run_multiseed.py

Run the benchmark across multiple random seeds and report mean ± std
for each policy. Produces Table 1 / Table 2 data for the paper.

Usage
-----
    python scripts/run_multiseed.py
    python scripts/run_multiseed.py --zipf-alpha 0.5 --output experiments/results/alpha05
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trajectorycache.evaluation.benchmark import run_benchmark, DEFAULT_POLICIES
from trajectorycache.simulation.runner import SimulationConfig
from trajectorycache.utils.config import load_config
from trajectorycache.utils.logging import setup_logging

# The 10 seeds reported in the paper (§4.4)
PAPER_SEEDS = [84810, 15592, 4278, 98196, 37048, 33098, 30256, 19289, 97530, 14434]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-seed TrajectoryCache benchmark")
    p.add_argument("--config", type=Path, default=Path("configs/simulation.yaml"))
    p.add_argument("--output", type=Path, default=Path("experiments/results"))
    p.add_argument("--zipf-alpha", type=float, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=PAPER_SEEDS)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level="WARNING")

    cfg_sumo = load_config(args.config)
    if args.zipf_alpha is not None:
        cfg_sumo.zipf_alpha = args.zipf_alpha
    cfg_sumo.platoon_size = 10

    cfg_simpy = load_config(args.config)
    if args.zipf_alpha is not None:
        cfg_simpy.zipf_alpha = args.zipf_alpha
    cfg_simpy.platoon_size = 1

    print(f"\n{'='*90}")
    print(f"Multi-seed benchmark | alpha={cfg_sumo.zipf_alpha} | seeds={args.seeds}")
    print(f"{'='*90}\n")

    class_names = {
        "trajectory": "TrajectoryCache",
        "lfu": "LFU",
        "lru": "LRU",
        "random": "Random",
        "fifo": "FIFO",
    }
    all_miss_sumo: dict[str, list[float]] = {
        class_names[name]: [] for name, _ in DEFAULT_POLICIES
    }
    all_miss_simpy: dict[str, list[float]] = {
        class_names[name]: [] for name, _ in DEFAULT_POLICIES
    }

    for i, seed in enumerate(args.seeds):
        cfg_sumo.seed = seed
        cfg_simpy.seed = seed
        print(f"Seed {i+1}/{len(args.seeds)}: {seed}")

        # SUMO
        print("  Running SUMO (platoons)... ", end="", flush=True)
        res_sumo = run_benchmark(config=cfg_sumo, verbose=args.verbose, output_dir=None)
        for name, metrics in res_sumo.items():
            if name in all_miss_sumo:
                all_miss_sumo[name].append(metrics.miss_rate * 100.0)
        tc_sumo = (
            res_sumo["TrajectoryCache"].miss_rate * 100.0
            if "TrajectoryCache" in res_sumo
            else 0.0
        )
        print(f"TC={tc_sumo:.2f}%")

        # SimPy
        print("  Running SimPy (indep)...   ", end="", flush=True)
        res_simpy = run_benchmark(config=cfg_simpy, verbose=args.verbose, output_dir=None)
        for name, metrics in res_simpy.items():
            if name in all_miss_simpy:
                all_miss_simpy[name].append(metrics.miss_rate * 100.0)
        tc_simpy = (
            res_simpy["TrajectoryCache"].miss_rate * 100.0
            if "TrajectoryCache" in res_simpy
            else 0.0
        )
        print(f"TC={tc_simpy:.2f}%")

    print(f"\n{'='*90}")
    print(
        f"RESULTS: alpha={cfg_sumo.zipf_alpha}  (mean miss rate % over {len(args.seeds)} seeds)"
    )
    print(f"{'='*90}")
    print(f"{'Policy':<18} | {'SimPy (Independent)':<20} | {'SUMO (Platoons)':<20}")
    print("-" * 65)

    summary_sumo = {}
    summary_simpy = {}
    for short_name, _ in DEFAULT_POLICIES:
        name = class_names[short_name]

        sim_v = np.array(all_miss_simpy[name])
        sumo_v = np.array(all_miss_sumo[name])

        sim_mean, sim_std = float(np.mean(sim_v)), float(np.std(sim_v))
        sumo_mean, sumo_std = float(np.mean(sumo_v)), float(np.std(sumo_v))

        display = "TrajectoryCache" if name == "TrajectoryCache" else name.upper()
        print(
            f"{display:<18} | {sim_mean:>6.2f} ± {sim_std:<5.2f}       | {sumo_mean:>6.2f} ± {sumo_std:<5.2f}"
        )

        summary_sumo[name] = {
            "miss_rate_mean": round(sumo_mean, 4),
            "miss_rate_std": round(sumo_std, 4),
            "per_seed": [round(float(v), 4) for v in sumo_v],
        }
        summary_simpy[name] = {
            "miss_rate_mean": round(sim_mean, 4),
            "miss_rate_std": round(sim_std, 4),
            "per_seed": [round(float(v), 4) for v in sim_v],
        }

    print(f"{'='*90}\n")

    # Save
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    fname = out / f"multiseed_alpha{cfg_sumo.zipf_alpha:.1f}.json"
    with open(fname, "w") as f:
        json.dump(
            {
                "zipf_alpha": cfg_sumo.zipf_alpha,
                "seeds": list(args.seeds),
                "sumo": summary_sumo,
                "simpy": summary_simpy,
            },
            f,
            indent=2,
        )
    print(f"Saved to {fname}")

    # AFTER writing results to disk — validate schema immediately
    REQUIRED_KEYS = {"sumo", "simpy"}
    REQUIRED_CONDITION_KEYS = {"TrajectoryCache", "LFU", "LRU", "FIFO", "Random"}
    REQUIRED_POLICY_KEYS = {"miss_rate_mean", "miss_rate_std", "per_seed"}
    N_SEEDS = len(args.seeds)

    with open(fname) as f:
        written = json.load(f)

    for cond in REQUIRED_KEYS:
        assert cond in written, f"JSON missing top-level key: '{cond}'"
        for policy in REQUIRED_CONDITION_KEYS:
            assert policy in written[cond], f"JSON missing policy '{policy}' under '{cond}'"
            for k in REQUIRED_POLICY_KEYS:
                assert k in written[cond][policy], f"JSON missing '{k}' for {cond}/{policy}"
            assert isinstance(
                written[cond][policy]["per_seed"], list
            ), f"per_seed must be a list for {cond}/{policy}"
            assert (
                len(written[cond][policy]["per_seed"]) == N_SEEDS
            ), f"per_seed length mismatch for {cond}/{policy}: expected {N_SEEDS}"

    print("[run_multiseed] JSON schema validated OK.")


if __name__ == "__main__":
    main()
