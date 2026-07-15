#!/usr/bin/env python
"""
scripts/run_benchmark.py

Run all cache policies under identical simulation conditions and
save results to JSON + optional PNG charts.

Usage
-----
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --config configs/simulation.yaml --output experiments/results
    python scripts/run_benchmark.py --n-steps 2000 --capacity 30 --seed 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trajectorycache.evaluation.benchmark import run_benchmark
from trajectorycache.evaluation.metrics import save_results
from trajectorycache.simulation.runner import SimulationConfig
from trajectorycache.utils.config import load_config
from trajectorycache.utils.logging import setup_logging
from trajectorycache.utils.plotting import plot_bar_comparison


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SpatialUrgencyCache (SU) policy benchmark")
    p.add_argument(
        "--config", type=Path, default=Path("configs/simulation.yaml"), help="YAML config file"
    )
    p.add_argument(
        "--output", type=Path, default=Path("experiments/results"), help="Output dir"
    )
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--capacity", type=int, default=None)
    p.add_argument("--n-vehicles", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--zipf-alpha", type=float, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--log-level", default="WARNING")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level)

    # Load base config then apply CLI overrides
    cfg: SimulationConfig = load_config(args.config)
    if args.n_steps is not None:
        cfg.n_steps = args.n_steps
    if args.capacity is not None:
        cfg.cache_capacity = args.capacity
    if args.n_vehicles is not None:
        cfg.n_vehicles = args.n_vehicles
    if args.seed is not None:
        cfg.seed = args.seed
    if args.zipf_alpha is not None:
        cfg.zipf_alpha = args.zipf_alpha

    print(
        f"Running benchmark | seed={cfg.seed} | n_steps={cfg.n_steps} | "
        f"capacity={cfg.cache_capacity} | n_vehicles={cfg.n_vehicles} | "
        f"zipf_alpha={cfg.zipf_alpha}"
    )

    results = run_benchmark(config=cfg, output_dir=args.output, verbose=args.verbose)

    # Save JSON output
    args.output.mkdir(parents=True, exist_ok=True)
    save_results(list(results.values()), args.output / "benchmark_results.json")

    # Bar chart
    hit_rates = {name: m.hit_rate for name, m in results.items()}
    plot_bar_comparison(
        metrics_dict=hit_rates,
        output_path=args.output / "hit_rate_comparison.png",
    )

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
