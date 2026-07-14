#!/usr/bin/env python
"""
Config ablation: find WHICH scenario knob makes the spatial-urgency policy (SU)
appear to beat LFU. Start from the configuration under which SU wins (the
paper's original 10 km setup) and change one knob at a time toward the realistic
535 m setup. The step that flips SU's SU-LFU margin from negative (wins) to
positive (loses) is the true artifact driver.

All runs: synthetic simulator, SU (urgency_weight=0.2) vs LFU, 10 seeds.
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

SEEDS = [84810, 15592, 4278, 98196, 37048, 33098, 30256, 19289, 97530, 14434]

# Ablation ladder: each step changes ONE knob from the previous.
# C0 = paper's original winning 10km config; C5 = realistic 535m config.
LADDER = [
    ("C0_orig_10km",        dict(road_length=10000, active_zone_length=1600, r_rel=800,
                                 unidirectional=False, n_vehicles=200, mean_speed=25.0)),
    ("C1_unidirectional",   dict(road_length=10000, active_zone_length=1600, r_rel=800,
                                 unidirectional=True,  n_vehicles=200, mean_speed=25.0)),
    ("C2_dispersed_content",dict(road_length=10000, active_zone_length=10000, r_rel=800,
                                 unidirectional=True,  n_vehicles=200, mean_speed=25.0)),
    ("C3_small_rrel",       dict(road_length=10000, active_zone_length=10000, r_rel=150,
                                 unidirectional=True,  n_vehicles=200, mean_speed=25.0)),
    ("C4_short_road",       dict(road_length=535,  active_zone_length=535,  r_rel=150,
                                 unidirectional=True,  n_vehicles=130, mean_speed=25.0)),
    ("C5_realistic_slow",   dict(road_length=535,  active_zone_length=535,  r_rel=150,
                                 unidirectional=True,  n_vehicles=130, mean_speed=7.7)),
]
POLICIES = [("LFU", "lfu", {"pop_window": 300.0}),
            ("SU", "trajectory", {"urgency_weight": 0.2})]
N_ITEMS, CAP, ZIPF = 200, 20, 0.8
WARMUP, MEASURE = 150, 600


def job(args):
    cname, cfgknobs, pname, pkey, kw, seed = args
    sys.path.insert(0, str(ROOT / "src"))
    from trajectorycache.cache import build_cache
    from trajectorycache.evaluation.metrics import compute_metrics
    from trajectorycache.simulation.runner import SimulationConfig, SimulationRunner

    cfg = SimulationConfig(n_items=N_ITEMS, zipf_alpha=ZIPF, cache_capacity=CAP,
                           n_steps=MEASURE, warmup_steps=WARMUP, speed_std=3.0,
                           platoon_size=10, seed=seed, **cfgknobs)
    mr = compute_metrics(SimulationRunner(build_cache(pkey, CAP, **kw), cfg).run()).miss_rate * 100.0
    return cname, pname, seed, mr


def main():
    t0 = time.time()
    jobs = [(cn, ck, pn, pk, kw, s) for (cn, ck) in LADDER
            for (pn, pk, kw) in POLICIES for s in SEEDS]
    raw = {}
    with Pool(processes=12) as pool:
        for cn, pn, seed, mr in pool.imap_unordered(job, jobs):
            raw.setdefault(cn, {}).setdefault(pn, {})[seed] = mr

    out = {}
    for cn, _ in LADDER:
        out[cn] = {p: {"mean": float(np.mean([raw[cn][p][s] for s in SEEDS])),
                       "std": float(np.std([raw[cn][p][s] for s in SEEDS])),
                       "per_seed": [round(raw[cn][p][s], 4) for s in SEEDS]}
                   for p in raw[cn]}
    out["_meta"] = {"seeds": SEEDS, "ladder": [c[0] for c in LADDER], "tier": "synthetic"}
    (ROOT / "experiments" / "results" / "config_ablation.json").write_text(json.dumps(out, indent=2))

    print("=== CONFIG ABLATION (synthetic; SU-LFU margin, negative = SU beats LFU) ===")
    print(f"{'config':22s}{'LFU':>8s}{'SU':>8s}{'SU-LFU':>9s}")
    prev = None
    for cn, _ in LADDER:
        lfu, su = out[cn]["LFU"]["mean"], out[cn]["SU"]["mean"]
        flip = ""
        if prev is not None and (prev < 0) != ((su - lfu) < 0):
            flip = "  <-- FLIP"
        print(f"{cn:22s}{lfu:8.2f}{su:8.2f}{su-lfu:+9.2f}{flip}")
        prev = su - lfu
    print(f"\nElapsed {time.time()-t0:.0f}s. Saved config_ablation.json")


if __name__ == "__main__":
    main()
