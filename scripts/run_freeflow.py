#!/usr/bin/env python
"""
Free-flow real-traffic replay (answers the "symmetry attack" review point).

The main paper evaluates on the congested NGSIM I-80 segment at a short request
radius (r_rel = 150 m). A reviewer can object that r_rel = 150 m is as
convenient in one direction as r_rel = 800 m is in the other, and that we only
tested a congested regime. Free-flow traffic is exactly the regime where a
*large* request radius is physically justified: fast vehicles cover more ground
within the policy's lookahead horizon (T_pred x speed), so if the spatial-urgency
signal ever helps on real data, it should help here.

This script runs the identical 8-policy harness on a free-flow real-trajectory
dataset and SWEEPS the request radius, so the SU-vs-LFU margin can be read as a
function of r_rel on genuinely free-flow data. It reuses the same NGSIM adapter
and demand model as run_real_ngsim.py, so mobility source is the only change.

DATA (not committed; download once):
  * NGSIM US-101 (has lighter-density / higher-speed windows) -- filter to the
    columns the adapter expects (vehicle_id, frame_id, global_time, local_y,
    v_vel, direction, lane_id) and save as:
        data/raw/ngsim/ngsim_us101_freeflow.csv
    Source: U.S. DOT open-data portal (NGSIM), dataset 8ect-6jqj.
  * OR highD (free-flow German highway). highD uses a different schema; convert
    a recording to the same 7-column CSV first (a highD->NGSIM shim is the only
    additional code needed; the adapter and this runner are already free-flow
    ready).

Usage:
    python scripts/run_freeflow.py --data data/raw/ngsim/ngsim_us101_freeflow.csv
    python scripts/run_freeflow.py --sanity        # 1 seed, LFU/SU/EDC, quick
"""
from __future__ import annotations

import argparse
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

DEFAULT_DATA = ROOT / "data" / "raw" / "ngsim" / "ngsim_us101_freeflow.csv"
N_ITEMS, CAP = 200, 20
WARMUP, MEASURE = 150, 600  # steps (1 Hz)
SEEDS = [84810, 15592, 4278, 98196, 37048, 33098, 30256, 19289, 97530, 14434]

# Read the SU-LFU margin as a function of radius on free-flow data. Radii are
# chosen for THIS segment (~669 m) and free-flow speed (~41 km/h => 11.4 m/s):
# the policy's lookahead distance is T_pred*speed = 30 s * 11.4 = ~340 m, so
# r_rel=350 m is the critical point where SU should win if it ever does.
#   150 m = deployment-realistic (short);
#   350 m = lookahead-matched (SU's best case in free-flow);
#   500 m = beyond lookahead but still < segment.
# (An 800 m radius exceeds the segment length and is degenerate here, so it is
#  intentionally excluded.)
R_SWEEP = [150.0, 350.0, 500.0]

POLICIES = [
    ("LRU", "lru", {}),
    ("FIFO", "fifo", {}),
    ("Random", "random", {}),
    ("LFU", "lfu", {"pop_window": 300.0}),
    ("Proximity", "proximity", {}),
    ("SU", "su", {"urgency_weight": 0.2}),
    ("EDC", "expected_demand", {}),
    ("QLearning", "qlearning", {"lr": 0.05}),
]


def _require_data(path: Path) -> None:
    if not path.exists():
        sys.exit(
            f"\n[run_freeflow] Free-flow dataset not found:\n    {path}\n\n"
            "This experiment needs a FREE-FLOW real-trajectory CSV in NGSIM\n"
            "format (columns: vehicle_id, frame_id, global_time, local_y,\n"
            "v_vel, direction, lane_id). See this file's module docstring for\n"
            "how to obtain NGSIM US-101 or convert highD. No synthetic stand-in\n"
            "is used on purpose: the result must come from real free-flow data.\n"
        )


# --- worker side (multiprocessing): the pre-parsed window is shared once. ---
_WINDOW = None  # set by the pool initializer


def _init_worker(window):
    global _WINDOW
    _WINDOW = window


def _run_one(task):
    """One independent (radius, policy, seed) replay -> miss rate (%)."""
    r, name, key, kw, seed, seg = task
    pk = dict(kw)
    if key in ("su", "proximity"):
        pk["r_rel"] = r
    if key == "expected_demand":
        pk["r_req"] = r
    cat = ContentCatalog(n_items=N_ITEMS, road_length=seg, active_zone_length=seg,
                         zipf_alpha=0.8, seed=seed)
    loc_map = cat.location_map()
    cache = build_cache(key, CAP, **pk)
    cache.clear()
    for i, (t, vehs) in enumerate(_WINDOW):
        if i == WARMUP:
            cache.reset_stats()
        for item in cat.generate_vehicle_requests(vehicles=vehs, r_request=r):
            cache.request(item_id=item.item_id, item_location=item.location,
                          current_time=t, vehicles=vehs, catalog=loc_map)
    return (f"{r:.0f}", name, seed, cache.summary()["miss_rate"])


def run(data_path: Path, seeds=SEEDS, policies=POLICIES, radii=R_SWEEP,
        jobs=None, verbose=True):
    from concurrent.futures import ProcessPoolExecutor
    import os

    steps, meta = load_ngsim_steps(data_path)
    seg = meta["segment_length_m"]
    window = steps[: WARMUP + MEASURE]
    jobs = jobs or os.cpu_count() or 4
    if verbose:
        print(
            f"Free-flow replay: seg={seg:.0f} m, {meta['n_vehicles']} vehicles, "
            f"mean {meta['mean_speed_mps'] * 3.6:.0f} km/h, mean on-road "
            f"{meta['mean_on_road']:.0f}, steps used {len(window)} | {jobs} workers",
            flush=True,
        )

    by_radius = {}
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker,
                             initargs=(window,)) as ex:
        for r in radii:  # radius-by-radius: gives incremental output + partial saves
            rstr = f"{r:.0f}"
            tasks = [(r, name, key, kw, seed, seg)
                     for (name, key, kw) in policies for seed in seeds]
            coll: dict[str, dict[int, float]] = {}
            for _, name, seed, miss in ex.map(_run_one, tasks, chunksize=1):
                coll.setdefault(name, {})[seed] = miss
            out = {}
            for name, key, kw in policies:
                per_seed = [coll[name][s] for s in seeds]
                out[name] = {
                    "mean": float(np.mean(per_seed)),
                    "std": float(np.std(per_seed)),
                    "per_seed": [round(float(v), 4) for v in per_seed],
                }
            margin = out["SU"]["mean"] - out["LFU"]["mean"]  # already percentages
            by_radius[rstr] = out
            if verbose:
                print(f"  [{time.time()-t_start:5.0f}s] r_rel={r:4.0f} m  "
                      f"SU-LFU = {margin:+.2f} pp  "
                      f"(SU {out['SU']['mean']:.2f}  LFU {out['LFU']['mean']:.2f}  "
                      f"EDC {out['EDC']['mean']:.2f})", flush=True)
            _save_partial(by_radius, meta, radii)
    return by_radius, meta


def _save_partial(by_radius, meta, radii):
    res = {
        "tier": "real_freeflow", "status": "partial" if len(by_radius) < len(radii) else "complete",
        "segment_m": meta["segment_length_m"], "mean_on_road": meta["mean_on_road"],
        "mean_speed_kmh": meta["mean_speed_mps"] * 3.6, "seeds": SEEDS,
        "radius_sweep_m": radii, "warmup": WARMUP, "measure": MEASURE,
        "by_radius": by_radius,
    }
    (ROOT / "experiments" / "results" / "real_freeflow.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Free-flow real-traffic caching replay")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA,
                    help="free-flow NGSIM-format CSV")
    ap.add_argument("--sanity", action="store_true",
                    help="1 seed, LFU/SU/EDC, single radius (quick check)")
    ap.add_argument("--policies", nargs="+", default=None,
                    help="subset of policy names to run, e.g. LFU SU EDC")
    args = ap.parse_args()
    _require_data(args.data)

    pols = POLICIES
    if args.policies:
        want = set(args.policies)
        pols = [p for p in POLICIES if p[0] in want]

    t0 = time.time()
    if args.sanity:
        pols = [p for p in POLICIES if p[0] in ("LFU", "SU", "EDC")]
        run(args.data, seeds=SEEDS[:1], policies=pols, radii=[150.0])
    else:
        run(args.data, policies=pols)  # saves real_freeflow.json incrementally
        print("saved real_freeflow.json")
    print(f"elapsed {time.time() - t0:.0f}s")
