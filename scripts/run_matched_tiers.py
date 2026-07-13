#!/usr/bin/env python
"""
Matched-scenario fidelity tiers: run all policies on SYNTHETIC (platoon) and
SUMO (Krauss) mobility at the SAME 535 m unidirectional geometry and ~matched
congested density as the real NGSIM i-80 tier, so the mobility source is the
only thing that changes across the fidelity ladder.

Scenario (identical to scripts/run_real_ngsim.py):
  segment 535 m, 200 items dispersed, cache 20, forward request window 150 m,
  Zipf alpha 0.8, warmup 150 + measure 600 steps at 1 Hz, 10 seeds.

Density is matched to NGSIM (~130 on-road). Because the Krauss fundamental
diagram differs from real i-80, matching density leaves SUMO at a lower
congested speed (~15 km/h vs NGSIM's ~28); this speed residual is reported,
not hidden. Synthetic mean speed is set to the NGSIM mean (7.7 m/s).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from multiprocessing import Pool

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SUMO_BIN = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe"
SUMO_DIR = ROOT / "sumo"
NET535B = SUMO_DIR / "highway535b.net.xml"   # 535 m + 2-lane@8 bottleneck

SEG = 535.0
N_ITEMS, CAP, R_REQ, ZIPF = 200, 20, 150.0, 0.8
WARMUP, MEASURE = 150, 600
SEEDS = [84810, 15592, 4278, 98196, 37048, 33098, 30256, 19289, 97530, 14434]
SUMO_VPH = 15000

POLICIES = [
    ("LRU", "lru", {}),
    ("FIFO", "fifo", {}),
    ("Random", "random", {}),
    ("LFU", "lfu", {"pop_window": 300.0}),
    ("Proximity", "proximity", {}),
    ("TC_W0.2", "trajectory", {"urgency_weight": 0.2}),
    ("EDC", "expected_demand", {"r_req": R_REQ}),
    ("QLearning", "qlearning", {"lr": 0.05}),
]


# ----------------------------- SYNTHETIC ---------------------------------
def synth_job(job):
    pname, pkey, kw, seed = job
    sys.path.insert(0, str(ROOT / "src"))
    from trajectorycache.cache import build_cache
    from trajectorycache.evaluation.metrics import compute_metrics
    from trajectorycache.simulation.runner import SimulationConfig, SimulationRunner

    cfg = SimulationConfig(
        road_length=SEG, active_zone_length=SEG, r_rel=R_REQ,
        n_vehicles=130, mean_speed=7.7, speed_std=3.0,
        platoon_size=10, unidirectional=True,
        n_items=N_ITEMS, zipf_alpha=ZIPF, cache_capacity=CAP,
        n_steps=MEASURE, warmup_steps=WARMUP, seed=seed,
    )
    cache = build_cache(pkey, CAP, **kw)
    mr = compute_metrics(SimulationRunner(cache=cache, config=cfg).run()).miss_rate * 100.0
    return "synthetic", pname, seed, mr


# ------------------------------- SUMO ------------------------------------
def gen_fcd_file(seed):
    """Generate one FCD file per seed ONCE (main process, sequential)."""
    routes = SUMO_DIR / f"mt_routes_{seed}.rou.xml"
    fcd = SUMO_DIR / f"mt_fcd_{seed}.xml"
    routes.write_text(
        f'<routes>\n'
        f'<vType id="car" carFollowModel="Krauss" maxSpeed="29.06" '
        f'speedFactor="normc(1,0.1,0.7,1.3)" minGap="2.5" accel="2.6" decel="4.5" sigma="0.5"/>\n'
        f'<route id="r" edges="AB BC"/>\n'
        f'<flow id="f" type="car" route="r" begin="0" end="900" vehsPerHour="{SUMO_VPH}" '
        f'departSpeed="max" departLane="random"/>\n</routes>\n')
    subprocess.run([SUMO_BIN, "-n", str(NET535B), "-r", str(routes),
                    "--fcd-output", str(fcd), "--step-length", "1",
                    "--begin", "0", "--end", "900", "--seed", str(seed),
                    "--no-step-log", "true", "--no-warnings", "true"],
                   check=True, capture_output=True)
    routes.unlink()
    return fcd


def _parse_fcd(seed):
    """Parse a pre-generated FCD file (worker; read-only, concurrency-safe)."""
    fcd = SUMO_DIR / f"mt_fcd_{seed}.xml"
    steps = []
    for ts in ET.parse(fcd).getroot():
        vehs = [{"x": float(v.get("x")), "speed": float(v.get("speed")), "direction": 1}
                for v in ts if float(v.get("x")) <= SEG]  # measure on AB only
        steps.append((float(ts.get("time")), vehs))
    return steps


def sumo_job(job):
    pname, pkey, kw, seed = job
    sys.path.insert(0, str(ROOT / "src"))
    from trajectorycache.cache import build_cache
    from trajectorycache.content.catalog import ContentCatalog

    steps = _parse_fcd(seed)
    window = steps[: WARMUP + MEASURE]
    cat = ContentCatalog(n_items=N_ITEMS, road_length=SEG, active_zone_length=SEG,
                         zipf_alpha=ZIPF, seed=seed)
    loc_map = cat.location_map()
    cache = build_cache(pkey, CAP, **kw)
    cache.clear()
    for i, (t, vehs) in enumerate(window):
        if i == WARMUP:
            cache.reset_stats()
        for item in cat.generate_vehicle_requests(vehicles=vehs, r_request=R_REQ):
            cache.request(item_id=item.item_id, item_location=item.location,
                          current_time=t, vehicles=vehs, catalog=loc_map)
    on_road = float(np.mean([len(v) for _, v in window]))
    return "sumo", pname, seed, cache.summary()["miss_rate"], on_road


def main():
    t0 = time.time()
    synth_jobs = [(n, k, kw, s) for (n, k, kw) in POLICIES for s in SEEDS]
    sumo_jobs = [(n, k, kw, s) for (n, k, kw) in POLICIES for s in SEEDS]

    # Pre-generate the 10 SUMO FCD files ONCE (sequential, avoids worker races).
    print("generating SUMO FCDs...", flush=True)
    for s in SEEDS:
        gen_fcd_file(s)

    raw = {"synthetic": {}, "sumo": {}}
    onroad = {"sumo": []}
    with Pool(processes=10) as pool:
        for tier, pname, seed, mr in pool.imap_unordered(synth_job, synth_jobs):
            raw[tier].setdefault(pname, {})[seed] = mr
        for res in pool.imap_unordered(sumo_job, sumo_jobs):
            tier, pname, seed, mr, orr = res
            raw[tier].setdefault(pname, {})[seed] = mr
            onroad["sumo"].append(orr)

    for s in SEEDS:  # cleanup FCD files
        (SUMO_DIR / f"mt_fcd_{s}.xml").unlink(missing_ok=True)

    def summ(tier):
        return {p: {"mean": float(np.mean([raw[tier][p][s] for s in SEEDS])),
                    "std": float(np.std([raw[tier][p][s] for s in SEEDS])),
                    "per_seed": [round(raw[tier][p][s], 4) for s in SEEDS]}
                for p in raw[tier]}

    out = {
        "synthetic": summ("synthetic"),
        "sumo": summ("sumo"),
        "_meta": {"segment_m": SEG, "seeds": SEEDS, "r_request_m": R_REQ,
                  "sumo_vph": SUMO_VPH, "sumo_mean_on_road": float(np.mean(onroad["sumo"])),
                  "synthetic_n_vehicles": 130, "warmup": WARMUP, "measure": MEASURE,
                  "note": "matched 535m unidirectional; density-matched to NGSIM ~130"}
    }
    (ROOT / "experiments" / "results" / "matched_tiers_535m.json").write_text(json.dumps(out, indent=2))

    print("=== MATCHED 535m TIERS (miss %, 10 seeds) ===")
    for tier in ["synthetic", "sumo"]:
        lfu = out[tier]["LFU"]["mean"]
        print(f"\n{tier} (SUMO on-road ~{out['_meta']['sumo_mean_on_road']:.0f})" if tier == "sumo" else f"\n{tier}:")
        for p, _, _ in POLICIES:
            d = out[tier][p]["mean"] - lfu
            print(f"  {p:12s} {out[tier][p]['mean']:6.2f} +/- {out[tier][p]['std']:.2f} ({d:+.2f} vs LFU)")
    print(f"\nElapsed {time.time()-t0:.0f}s. Saved matched_tiers_535m.json")


if __name__ == "__main__":
    main()
