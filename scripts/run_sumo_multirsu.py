#!/usr/bin/env python
"""
Multi-RSU SUMO validation: test whether distributing several RSUs along the
highway (smaller per-RSU coverage -> lower effective per-zone density) restores
TC's advantage over LFU on realistic Krauss traffic, where a single RSU did not.

Pre-committed realistic config (NOT tuned to win):
  5 RSUs at x = 1000,3000,5000,7000,9000 m (every 2 km)
  each covers +/- 500 m, each an independent cache of capacity 20.
Each RSU serves the vehicles currently in its coverage; miss rate aggregated
over all RSUs. Same SUMO traffic drives every policy.
"""
import sys, subprocess, time, json
import xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from trajectorycache.cache import build_cache
from trajectorycache.content.catalog import ContentCatalog

ROOT = Path(__file__).resolve().parents[1]
SUMO_BIN = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe"
SUMO_DIR = ROOT / "sumo"
NET = SUMO_DIR / "highway.net.xml"
ROAD_LEN, ACTIVE_ZONE, N_ITEMS, CAP = 10000.0, 1600.0, 200, 20
WARMUP, MEASURE = 100, 300
SEEDS = [84810, 15592, 4278, 98196, 37048]
RSU_POS = [1000.0, 3000.0, 5000.0, 7000.0, 9000.0]
R_COV = 500.0
R_REL = 800.0

def write_routes(path, vph, end):
    path.write_text(f'''<routes>
    <vType id="car" carFollowModel="Krauss" maxSpeed="33.33" speedFactor="normc(1,0.1,0.7,1.3)" minGap="2.5" accel="2.6" decel="4.5" sigma="0.5"/>
    <route id="rAB" edges="AB"/>
    <route id="rBA" edges="BA"/>
    <flow id="fAB" type="car" route="rAB" begin="0" end="{end}" vehsPerHour="{vph}"/>
    <flow id="fBA" type="car" route="rBA" begin="0" end="{end}" vehsPerHour="{vph}"/>
</routes>''')

def run_sumo(routes, fcd, end, seed):
    subprocess.run([SUMO_BIN, "-n", str(NET), "-r", str(routes), "--fcd-output", str(fcd),
                    "--step-length", "1", "--begin", "0", "--end", str(end), "--seed", str(seed),
                    "--no-step-log", "true", "--no-warnings", "true"], check=True, capture_output=True)

def parse_fcd(fcd):
    steps = []
    for ts in ET.parse(fcd).getroot():
        vehs = []
        for v in ts:
            ang = float(v.get("angle"))
            vehs.append({"x": float(v.get("x")), "speed": float(v.get("speed")),
                         "direction": 1 if ang < 180 else -1})
        steps.append((float(ts.get("time")), vehs))
    return steps

def replay_multirsu(steps, policy, kw, seed):
    cat = ContentCatalog(n_items=N_ITEMS, road_length=ROAD_LEN, active_zone_length=ACTIVE_ZONE,
                         zipf_alpha=0.8, seed=seed)
    loc_map = cat.location_map()
    # one independent cache per RSU
    rsus = []
    for pos in RSU_POS:
        kw2 = dict(kw)
        if policy == "adaptive":
            kw2["cache_center"] = pos
        rsus.append(build_cache(policy, CAP, **kw2))
    for c in rsus:
        c.clear()
    start = next((i for i,(t,vs) in enumerate(steps) if len(vs) >= 20), 0)
    window = steps[start: start + WARMUP + MEASURE]
    for i,(t,vehs) in enumerate(window):
        if i == WARMUP:
            for c in rsus: c.reset_stats()
        for k, pos in enumerate(RSU_POS):
            local = [v for v in vehs if abs(v["x"] - pos) <= R_COV]
            if not local:
                continue
            reqs = cat.generate_vehicle_requests(vehicles=local, r_request=R_REL)
            for item in reqs:
                rsus[k].request(item_id=item.item_id, item_location=item.location,
                                current_time=t, vehicles=local, catalog=loc_map)
    hits = sum(c.summary()["hits"] for c in rsus)
    miss = sum(c.summary()["misses"] for c in rsus)
    return 100.0 * miss / (hits + miss) if (hits+miss) else 0.0

def main():
    t0 = time.time()
    scenarios = {"moderate": 1100, "high": 2600}
    out = {}
    for label, vph in scenarios.items():
        res = {p: [] for p in ["LFU","FixedW0.2","AdaptiveTC"]}
        for seed in SEEDS:
            routes = SUMO_DIR / f"mr_{label}_{seed}.rou.xml"
            fcd = SUMO_DIR / f"mr_{label}_{seed}.fcd.xml"
            write_routes(routes, vph, 600); run_sumo(routes, fcd, 600, seed)
            steps = parse_fcd(fcd)
            res["LFU"].append(replay_multirsu(steps, "lfu", {}, seed))
            res["FixedW0.2"].append(replay_multirsu(steps, "trajectory", {"urgency_weight":0.2}, seed))
            res["AdaptiveTC"].append(replay_multirsu(steps, "adaptive", {"w_max":0.2}, seed))
            fcd.unlink(); routes.unlink()
        m = {p:(float(np.mean(res[p])), float(np.std(res[p]))) for p in res}
        out[label] = {p:{"mean":m[p][0],"std":m[p][1]} for p in m}
        lfu,fx,ad = m["LFU"][0],m["FixedW0.2"][0],m["AdaptiveTC"][0]
        print(f"[{label}] 5xRSU  LFU={lfu:.2f}  Fixed={fx:.2f}({fx-lfu:+.2f})  Adaptive={ad:.2f}({ad-lfu:+.2f})")
    out["_meta"] = {"seeds":SEEDS,"rsu_positions":RSU_POS,"r_cov":R_COV,"elapsed_s":time.time()-t0,
                    "source":"SUMO 1.27 Krauss FCD, 5 RSU"}
    (ROOT/"experiments"/"results"/"sumo_multirsu.json").write_text(json.dumps(out, indent=2))
    print(f"Total elapsed: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
