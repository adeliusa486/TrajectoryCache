#!/usr/bin/env python
"""
Multi-RSU SUMO validation grid (EDC edition).

Extends the original single-scenario run_sumo_multirsu.py to the same
geography x traffic grid used for the single-RSU EDC validation, and adds the
ExpectedDemandCache (EDC) policy alongside LFU and the original fixed-weight
SpatialUrgencyCache (SU). Five independent RSUs (each an independent cache of capacity
20) are distributed every 2 km; miss rate is aggregated over all RSUs. Every
policy replays the SAME SUMO Krauss traffic per (geometry, traffic, seed).

Two content geographies (matching the single-RSU grid):
  - hotspot1600  : all items in a 1600 m active zone at road centre
  - corridor10000: items dispersed over the full 10 km road

This is a characterization grid, NOT tuned to win: results are reported for
every cell regardless of outcome. Per-seed values are stored in the output
JSON so every reported mean is auditable.
"""
import sys, subprocess, time, json
import xml.etree.ElementTree as ET
from pathlib import Path
from multiprocessing import Pool

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SUMO_BIN = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe"
SUMO_DIR = ROOT / "sumo"
NET = SUMO_DIR / "highway.net.xml"

ROAD_LEN, N_ITEMS, CAP = 10000.0, 200, 20
WARMUP, MEASURE = 100, 300
SEEDS = [84810, 15592, 4278, 98196, 37048, 33098, 30256, 19289, 97530, 14434]
RSU_POS = [1000.0, 3000.0, 5000.0, 7000.0, 9000.0]
R_COV = 500.0
R_REL = 800.0

POLICIES = [
    ("LFU", "lfu", {}),
    ("TC_W0.2", "trajectory", {"urgency_weight": 0.2}),
    ("EDC", "expected_demand", {}),
]
GEOMETRIES = [("hotspot1600", 1600.0), ("corridor10000", 10000.0)]
TRAFFIC = [("sparse", 150), ("moderate", 1100), ("high", 2600)]


def write_routes(path, vph, end):
    path.write_text(f'''<routes>
    <vType id="car" carFollowModel="Krauss" maxSpeed="33.33" speedFactor="normc(1,0.1,0.7,1.3)" minGap="2.5" accel="2.6" decel="4.5" sigma="0.5"/>
    <route id="rAB" edges="AB"/>
    <route id="rBA" edges="BA"/>
    <flow id="fAB" type="car" route="rAB" begin="0" end="{end}" vehsPerHour="{vph}"/>
    <flow id="fBA" type="car" route="rBA" begin="0" end="{end}" vehsPerHour="{vph}"/>
</routes>''')


def one_job(args):
    geo_label, zone, traffic_label, vph, seed = args
    sys.path.insert(0, str(ROOT / "src"))
    from trajectorycache.cache import build_cache
    from trajectorycache.content.catalog import ContentCatalog

    tag = f"{geo_label}_{traffic_label}_{seed}"
    routes = SUMO_DIR / f"mrg_routes_{tag}.rou.xml"
    fcd = SUMO_DIR / f"mrg_fcd_{tag}.xml"
    write_routes(routes, vph, end=600)
    subprocess.run([SUMO_BIN, "-n", str(NET), "-r", str(routes),
                    "--fcd-output", str(fcd), "--step-length", "1",
                    "--begin", "0", "--end", "600", "--seed", str(seed),
                    "--no-step-log", "true", "--no-warnings", "true"],
                   check=True, capture_output=True)
    steps = []
    for ts in ET.parse(fcd).getroot():
        vehs = [{"x": float(v.get("x")), "speed": float(v.get("speed")),
                 "direction": 1 if float(v.get("angle")) < 180 else -1}
                for v in ts]
        steps.append((float(ts.get("time")), vehs))
    routes.unlink(); fcd.unlink()

    cat = ContentCatalog(n_items=N_ITEMS, road_length=ROAD_LEN,
                         active_zone_length=zone, zipf_alpha=0.8, seed=seed)
    loc_map = cat.location_map()

    start = next((i for i, (t, vs) in enumerate(steps) if len(vs) >= 20), 0)
    window = steps[start:start + WARMUP + MEASURE]

    # Pre-generate per-RSU requests ONCE so every policy sees identical demand.
    # For each step: list of (rsu_index, local_vehicles, [(item_id, loc), ...]).
    per_step = []
    for (t, vehs) in window:
        rsu_reqs = []
        for k, pos in enumerate(RSU_POS):
            local = [v for v in vehs if abs(v["x"] - pos) <= R_COV]
            if not local:
                continue
            reqs = cat.generate_vehicle_requests(vehicles=local, r_request=R_REL)
            rsu_reqs.append((k, local, [(it.item_id, it.location) for it in reqs]))
        per_step.append((t, rsu_reqs))

    on_road = float(np.mean([len(vs) for _, vs in window]))

    out = {}
    for name, key, kw in POLICIES:
        rsus = [build_cache(key, CAP, **kw) for _ in RSU_POS]
        for c in rsus:
            c.clear()
        for i, (t, rsu_reqs) in enumerate(per_step):
            if i == WARMUP:
                for c in rsus:
                    c.reset_stats()
            for (k, local, reqs) in rsu_reqs:
                for iid, loc in reqs:
                    rsus[k].request(item_id=iid, item_location=loc, current_time=t,
                                    vehicles=local, catalog=loc_map)
        hits = sum(c.summary()["hits"] for c in rsus)
        miss = sum(c.summary()["misses"] for c in rsus)
        out[name] = 100.0 * miss / (hits + miss) if (hits + miss) else float("nan")

    return geo_label, traffic_label, seed, on_road, out


def main():
    t0 = time.time()
    jobs = [(g, zone, tl, vph, s)
            for (g, zone) in GEOMETRIES
            for (tl, vph) in TRAFFIC
            for s in SEEDS]

    results = {}
    with Pool(processes=10) as pool:
        for k, (geo, tr, seed, onroad, out) in enumerate(pool.imap_unordered(one_job, jobs), 1):
            results.setdefault(geo, {}).setdefault(tr, {"on_road": [], "runs": {}})
            results[geo][tr]["on_road"].append(onroad)
            for pol, mr in out.items():
                results[geo][tr]["runs"].setdefault(pol, []).append(mr)
            print(f"[{k}/{len(jobs)}] {geo}/{tr} seed={seed} onroad~{onroad:.0f} " +
                  " ".join(f"{p}={m:.2f}" for p, m in out.items()), flush=True)

    summary = {}
    for geo in results:
        summary[geo] = {}
        for tr in results[geo]:
            runs = results[geo][tr]["runs"]
            summary[geo][tr] = {
                "on_road_mean": float(np.mean(results[geo][tr]["on_road"])),
                **{p: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                       "per_seed": [float(x) for x in v]}
                   for p, v in runs.items()},
            }
    summary["_meta"] = {"seeds": SEEDS, "rsu_positions": RSU_POS, "r_cov": R_COV,
                        "elapsed_s": time.time() - t0,
                        "source": "SUMO 1.27 Krauss FCD, 5 RSU",
                        "policies": [p[0] for p in POLICIES]}
    out_path = ROOT / "experiments" / "results" / "sumo_multirsu_edc_10seed.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print("\n==== MULTI-RSU SUMMARY (miss rate %, mean over 10 seeds) ====")
    for geo in ["hotspot1600", "corridor10000"]:
        for tr in ["sparse", "moderate", "high"]:
            s = summary[geo][tr]
            lfu = s["LFU"]["mean"]
            line = f"{geo:14s} {tr:9s} onroad~{s['on_road_mean']:.0f}  "
            for p in ["LFU", "TC_W0.2", "EDC"]:
                line += f"{p}={s[p]['mean']:.2f}({s[p]['mean']-lfu:+.2f}) "
            print(line)
    print(f"\nElapsed {time.time()-t0:.0f}s. Saved to {out_path}")


if __name__ == "__main__":
    main()
