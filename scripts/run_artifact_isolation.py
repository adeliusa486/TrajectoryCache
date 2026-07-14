#!/usr/bin/env python
"""
N5 -- Artifact isolation (the paper's centerpiece).

Holds the traffic SOURCE fixed (synthetic simulator) and the road geometry
fixed (535 m unidirectional), and sweeps the scenario-CONFIGURATION knobs that
prior spatial-caching evaluations vary freely:
  - platoon_size   : 1 (no artificial clustering) vs 10 (platoon clustering)
  - n_vehicles     : 30 (sparse) vs 130 (dense, matches NGSIM i-80)
  - r_rel / r_req  : 150 m vs 400 m (forward request window)

For each configuration cell we report the SU-minus-LFU and EDC-minus-LFU miss-
rate margins (negative = beats LFU), 10 seeds, per-seed stored. The thesis
prediction: SU beats LFU ONLY in the platoon-clustered + sparse cells; under
the dense / no-platoon cells (which resemble real congested traffic) SU loses.
This localizes the "spatial-urgency gain" to specific configuration choices
rather than to any property of the traffic itself.
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

SEG = 535.0
N_ITEMS, CAP, ZIPF = 200, 20, 0.8
WARMUP, MEASURE = 150, 600
SEEDS = [84810, 15592, 4278, 98196, 37048, 33098, 30256, 19289, 97530, 14434]

PLATOON = [1, 10]
DENSITY = [30, 130]
RREL = [150.0, 400.0]
POLICIES = [("LFU", "lfu", {"pop_window": 300.0}),
            ("SU", "trajectory", {"urgency_weight": 0.2}),
            ("EDC", "expected_demand", {})]


def job(args):
    platoon, nveh, rrel, pname, pkey, kw, seed = args
    sys.path.insert(0, str(ROOT / "src"))
    from trajectorycache.cache import build_cache
    from trajectorycache.evaluation.metrics import compute_metrics
    from trajectorycache.simulation.runner import SimulationConfig, SimulationRunner

    kw = dict(kw)
    if pkey == "expected_demand":
        kw["r_req"] = rrel
    cfg = SimulationConfig(
        road_length=SEG, active_zone_length=SEG, r_rel=rrel,
        n_vehicles=nveh, mean_speed=7.7, speed_std=3.0,
        platoon_size=platoon, unidirectional=True,
        n_items=N_ITEMS, zipf_alpha=ZIPF, cache_capacity=CAP,
        n_steps=MEASURE, warmup_steps=WARMUP, seed=seed,
    )
    mr = compute_metrics(SimulationRunner(build_cache(pkey, CAP, **kw), cfg).run()).miss_rate * 100.0
    cell = f"platoon{platoon}_n{nveh}_r{int(rrel)}"
    return cell, pname, seed, mr


def main():
    t0 = time.time()
    jobs = [(p, n, r, pn, pk, kw, s)
            for p in PLATOON for n in DENSITY for r in RREL
            for (pn, pk, kw) in POLICIES for s in SEEDS]

    raw = {}
    with Pool(processes=12) as pool:
        for cell, pname, seed, mr in pool.imap_unordered(job, jobs):
            raw.setdefault(cell, {}).setdefault(pname, {})[seed] = mr

    out = {}
    for cell in raw:
        out[cell] = {}
        for pname in raw[cell]:
            vals = [raw[cell][pname][s] for s in SEEDS]
            out[cell][pname] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                                "per_seed": [round(v, 4) for v in vals]}
    out["_meta"] = {"segment_m": SEG, "seeds": SEEDS, "tier": "synthetic",
                    "swept": {"platoon": PLATOON, "n_vehicles": DENSITY, "r_rel": RREL}}
    (ROOT / "experiments" / "results" / "artifact_isolation.json").write_text(json.dumps(out, indent=2))

    print("=== ARTIFACT ISOLATION (synthetic, 535m; SU-LFU and EDC-LFU margin, pp) ===")
    print(f"{'config':22s}{'LFU':>8s}{'SU':>8s}{'SU-LFU':>9s}{'EDC-LFU':>9s}")
    for p in PLATOON:
        for n in DENSITY:
            for r in RREL:
                cell = f"platoon{p}_n{n}_r{int(r)}"
                c = out[cell]
                lfu = c["LFU"]["mean"]
                print(f"{cell:22s}{lfu:8.2f}{c['SU']['mean']:8.2f}"
                      f"{c['SU']['mean']-lfu:+9.2f}{c['EDC']['mean']-lfu:+9.2f}")
    print(f"\nElapsed {time.time()-t0:.0f}s. Saved artifact_isolation.json")


if __name__ == "__main__":
    main()
