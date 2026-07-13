#!/usr/bin/env python
"""
Synthetic-simulator suite with ExpectedDemandCache (EDC) added.

Re-runs the paper's synthetic benchmark (the in-house highway simulator, NOT
SUMO) across the 10-seed protocol for every policy plus EDC, under both the
platoon ("SUMO"-labelled, platoon_size=10) and independent ("SimPy",
platoon_size=1) demand conditions, at Zipf alpha in {0.8, 0.5}. Also runs the
vehicle-density sweep with EDC (5-seed subset, matching the paper). Per-seed
values are stored so every mean is auditable.

Parallelized across a flat job list. QLearning is intentionally excluded here
(it is ~25x slower and its numbers are reported separately); this suite
provides the EDC column for the paper's rule-based/classical tables.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CONFIG = ROOT / "configs" / "simulation.yaml"
OUTDIR = ROOT / "experiments" / "results"
SEEDS = [84810, 15592, 4278, 98196, 37048, 33098, 30256, 19289, 97530, 14434]
SEEDS5 = SEEDS[:5]

POLICIES = [
    ("LRU", "lru", {}),
    ("FIFO", "fifo", {}),
    ("Random", "random", {}),
    ("LFU", "lfu", {"pop_window": 300.0}),
    ("Proximity", "proximity", {}),
    ("TrajectoryCache", "trajectory", {"urgency_weight": 0.2}),
    ("EDC", "expected_demand", {}),
]


def run_one(job):
    """job = (cell_key, policy_name, policy_key, kw, seed, alpha, platoon, extra)"""
    (cell_key, pname, pkey, kw, seed, alpha, platoon, extra) = job
    sys.path.insert(0, str(ROOT / "src"))
    from trajectorycache.cache import build_cache
    from trajectorycache.evaluation.metrics import compute_metrics
    from trajectorycache.simulation.runner import SimulationRunner
    from trajectorycache.utils.config import load_config

    cfg = load_config(CONFIG)
    cfg.seed = seed
    cfg.zipf_alpha = alpha
    cfg.platoon_size = platoon
    if extra:
        for k, v in extra.items():
            setattr(cfg, k, v)
    cache = build_cache(pkey, cfg.cache_capacity, **kw)
    runner = SimulationRunner(cache=cache, config=cfg)
    mr = compute_metrics(runner.run()).miss_rate * 100.0
    return cell_key, pname, seed, mr


def build_jobs():
    jobs = []
    # Main policy tables: alpha in {0.8, 0.5} x {platoon, independent}
    for alpha in [0.8, 0.5]:
        for cond_label, platoon in [("platoon", 10), ("independent", 1)]:
            cell = f"alpha{alpha}/{cond_label}"
            for pname, pkey, kw in POLICIES:
                for s in SEEDS:
                    jobs.append((cell, pname, pkey, kw, s, alpha, platoon, None))
    # Density sweep (5-seed subset), alpha=0.8, platoon
    for n in [50, 100, 200, 400, 600]:
        cell = f"density/{n}"
        for pname, pkey, kw in [("LFU", "lfu", {"pop_window": 300.0}),
                                ("TrajectoryCache", "trajectory", {"urgency_weight": 0.2}),
                                ("EDC", "expected_demand", {})]:
            for s in SEEDS5:
                jobs.append((cell, pname, pkey, kw, s, 0.8, 10, {"n_vehicles": n}))
    return jobs


def main():
    t0 = time.time()
    jobs = build_jobs()
    print(f"Total jobs: {len(jobs)}", flush=True)

    raw = {}  # cell -> policy -> {seed: mr}
    with Pool(processes=12) as pool:
        for k, (cell, pname, seed, mr) in enumerate(pool.imap_unordered(run_one, jobs), 1):
            raw.setdefault(cell, {}).setdefault(pname, {})[seed] = mr
            if k % 40 == 0:
                print(f"  {k}/{len(jobs)} done", flush=True)

    def summarize(cell, seeds):
        out = {}
        for pname, d in raw[cell].items():
            vals = [d[s] for s in seeds]
            out[pname] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                          "per_seed": [round(float(v), 4) for v in vals]}
        return out

    results = {}
    for alpha in [0.8, 0.5]:
        results[f"alpha{alpha}"] = {
            "platoon": summarize(f"alpha{alpha}/platoon", SEEDS),
            "independent": summarize(f"alpha{alpha}/independent", SEEDS),
        }
    results["density_sweep_alpha0.8"] = {
        str(n): summarize(f"density/{n}", SEEDS5) for n in [50, 100, 200, 400, 600]
    }
    results["_meta"] = {"seeds": SEEDS, "seeds5": SEEDS5, "elapsed_s": time.time() - t0,
                        "simulator": "in-house synthetic highway (NOT SUMO)",
                        "policies": [p[0] for p in POLICIES]}

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTDIR / "synthetic_edc_10seed.json"
    out_path.write_text(json.dumps(results, indent=2))

    print("\n==== SYNTHETIC SUITE SUMMARY (miss rate %, 10 seeds) ====")
    for alpha in [0.8, 0.5]:
        for cond in ["platoon", "independent"]:
            t = results[f"alpha{alpha}"][cond]
            lfu = t["LFU"]["mean"]
            print(f"\nalpha={alpha} {cond}:")
            for pname, _, _ in POLICIES:
                d = t[pname]["mean"] - lfu
                print(f"  {pname:16s} {t[pname]['mean']:6.2f} +/- {t[pname]['std']:.2f}"
                      f"  ({d:+.2f} vs LFU)")
    print("\ndensity sweep (5 seeds, alpha=0.8):")
    for n in [50, 100, 200, 400, 600]:
        c = results["density_sweep_alpha0.8"][str(n)]
        print(f"  n={n:3d}  LFU={c['LFU']['mean']:.2f}  TC={c['TrajectoryCache']['mean']:.2f}  "
              f"EDC={c['EDC']['mean']:.2f}  (EDC vs LFU {c['EDC']['mean']-c['LFU']['mean']:+.2f})")
    print(f"\nElapsed {time.time()-t0:.0f}s. Saved to {out_path}")


if __name__ == "__main__":
    main()
