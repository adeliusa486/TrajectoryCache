#!/usr/bin/env python
"""
Tier-3 (real trajectories) replay: run cache policies on NGSIM i-80 vehicle
trajectories, using the identical demand model and replay logic as the SUMO
tiers so that mobility source is the only thing that changes.

Scenario is scaled to the NGSIM segment (~535 m). Content is dispersed over the
segment; a single cache serves all in-segment vehicles. The trajectory is fixed
(real data); the stochastic element across seeds is the catalog item placement
and Zipf demand draws, matching how seeds are used in the other tiers.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from adapters.ngsim_adapter import load_ngsim_steps
from trajectorycache.cache import build_cache
from trajectorycache.content.catalog import ContentCatalog

NGSIM_CSV = ROOT / "data" / "raw" / "ngsim" / "ngsim_i80_win1.csv"
N_ITEMS, CAP = 200, 20
R_REQUEST = 150.0          # forward request window (m), scaled to short segment
WARMUP, MEASURE = 150, 600  # steps (1 Hz)
SEEDS = [84810, 15592, 4278, 98196, 37048, 33098, 30256, 19289, 97530, 14434]

POLICIES = [
    ("LRU", "lru", {}),
    ("FIFO", "fifo", {}),
    ("Random", "random", {}),
    ("LFU", "lfu", {"pop_window": 300.0}),
    ("Proximity", "proximity", {}),
    ("TC_W0.2", "trajectory", {"urgency_weight": 0.2}),
    ("EDC", "expected_demand", {"r_req": R_REQUEST}),
    ("QLearning", "qlearning", {"lr": 0.05}),
]


def run(seeds=SEEDS, policies=POLICIES, verbose=True):
    steps, meta = load_ngsim_steps(NGSIM_CSV)
    seg = meta["segment_length_m"]
    window = steps[: WARMUP + MEASURE]
    if verbose:
        print(f"NGSIM i-80: seg={seg:.0f} m, {meta['n_vehicles']} vehicles, "
              f"mean {meta['mean_speed_mps']*3.6:.0f} km/h, mean on-road "
              f"{meta['mean_on_road']:.0f}, steps used {len(window)}", flush=True)

    out = {}
    for name, key, kw in policies:
        per_seed = []
        for seed in seeds:
            cat = ContentCatalog(n_items=N_ITEMS, road_length=seg,
                                 active_zone_length=seg, zipf_alpha=0.8, seed=seed)
            loc_map = cat.location_map()
            cache = build_cache(key, CAP, **kw)
            cache.clear()
            for i, (t, vehs) in enumerate(window):
                if i == WARMUP:
                    cache.reset_stats()
                for item in cat.generate_vehicle_requests(vehicles=vehs, r_request=R_REQUEST):
                    cache.request(item_id=item.item_id, item_location=item.location,
                                  current_time=t, vehicles=vehs, catalog=loc_map)
            per_seed.append(cache.summary()["miss_rate"])
        out[name] = {"mean": float(np.mean(per_seed)), "std": float(np.std(per_seed)),
                     "per_seed": [round(float(v), 4) for v in per_seed]}
        if verbose:
            print(f"  {name:12s} {out[name]['mean']:6.2f} +/- {out[name]['std']:.2f}", flush=True)
    return out, meta


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true", help="1 seed, LFU/TC/EDC only")
    args = ap.parse_args()
    t0 = time.time()
    if args.sanity:
        pols = [p for p in POLICIES if p[0] in ("LFU", "TC_W0.2", "EDC")]
        out, meta = run(seeds=SEEDS[:1], policies=pols)
    else:
        out, meta = run()
        res = {"tier": "real_ngsim_i80", "segment_m": meta["segment_length_m"],
               "mean_on_road": meta["mean_on_road"],
               "mean_speed_kmh": meta["mean_speed_mps"] * 3.6,
               "seeds": SEEDS, "r_request_m": R_REQUEST,
               "warmup": WARMUP, "measure": MEASURE, "policies": out}
        (ROOT / "experiments" / "results" / "real_ngsim_i80.json").write_text(json.dumps(res, indent=2))
        print("saved real_ngsim_i80.json")
    print(f"elapsed {time.time()-t0:.0f}s")
