#!/usr/bin/env python
"""
SUMO validation (#3): replay genuine Krauss car-following traffic through the
caches, to confirm the synthetic-model findings (TC > LFU, Adaptive >= Fixed)
hold under realistic micro-mobility rather than the simplified platoon model.

Pipeline: generate flows -> run SUMO with FCD output -> parse per-step vehicle
states -> drive ContentCatalog demand + each cache policy on the SAME stream.
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

def write_routes(path, veh_per_hour, end):
    path.write_text(f'''<routes>
    <vType id="car" carFollowModel="Krauss" maxSpeed="33.33" speedFactor="normc(1,0.1,0.7,1.3)" minGap="2.5" accel="2.6" decel="4.5" sigma="0.5"/>
    <route id="rAB" edges="AB"/>
    <route id="rBA" edges="BA"/>
    <flow id="fAB" type="car" route="rAB" begin="0" end="{end}" vehsPerHour="{veh_per_hour}"/>
    <flow id="fBA" type="car" route="rBA" begin="0" end="{end}" vehsPerHour="{veh_per_hour}"/>
</routes>''')

def run_sumo(routes, fcd, end, seed):
    subprocess.run([SUMO_BIN, "-n", str(NET), "-r", str(routes),
                    "--fcd-output", str(fcd), "--step-length", "1",
                    "--begin", "0", "--end", str(end), "--seed", str(seed),
                    "--no-step-log", "true", "--no-warnings", "true"],
                   check=True, capture_output=True)

def parse_fcd(fcd):
    """Return list of (t, [ {x,speed,direction}, ... ]) per timestep."""
    steps = []
    for ts in ET.parse(fcd).getroot():
        vehs = []
        for v in ts:
            ang = float(v.get("angle"))
            direction = 1 if (ang < 180) else -1   # 90=east(+x), 270=west(-x)
            vehs.append({"x": float(v.get("x")), "speed": float(v.get("speed")),
                         "direction": direction})
        steps.append((float(ts.get("time")), vehs))
    return steps

def replay(steps, policy, kw, seed):
    cat = ContentCatalog(n_items=N_ITEMS, road_length=ROAD_LEN,
                         active_zone_length=ACTIVE_ZONE, zipf_alpha=0.8, seed=seed)
    loc_map = cat.location_map()
    cache = build_cache(policy, CAP, **kw)
    cache.clear()
    # use measurement window after enough vehicles have entered
    start = next((i for i,(t,vs) in enumerate(steps) if len(vs) >= 20), 0)
    window = steps[start: start + WARMUP + MEASURE]
    for i,(t,vehs) in enumerate(window):
        if i == WARMUP:
            cache.reset_stats()
        reqs = cat.generate_vehicle_requests(vehicles=vehs, r_request=800.0)
        for item in reqs:
            cache.request(item_id=item.item_id, item_location=item.location,
                          current_time=t, vehicles=vehs, catalog=loc_map)
    s = cache.summary()
    return s["miss_rate"], np.mean([len(vs) for _,vs in window])

def main():
    t0 = time.time()
    # vehsPerHour tuned to give moderate (~200) and high (~450) on-road counts
    scenarios = {"moderate": 1100, "high": 2600}
    out = {}
    for label, vph in scenarios.items():
        out[label] = {p: [] for p in ["LFU","FixedW0.2","AdaptiveTC"]}
        onroad = []
        for seed in SEEDS:
            routes = SUMO_DIR / f"routes_{label}_{seed}.rou.xml"
            fcd = SUMO_DIR / f"fcd_{label}_{seed}.xml"
            write_routes(routes, vph, end=600)
            run_sumo(routes, fcd, end=600, seed=seed)
            steps = parse_fcd(fcd)
            mr_lfu,_ = replay(steps, "lfu", {}, seed)
            mr_fix,_ = replay(steps, "trajectory", {"urgency_weight":0.2}, seed)
            mr_ad, n = replay(steps, "adaptive", {"w_max":0.2}, seed)
            out[label]["LFU"].append(mr_lfu)
            out[label]["FixedW0.2"].append(mr_fix)
            out[label]["AdaptiveTC"].append(mr_ad)
            onroad.append(n)
            fcd.unlink(); routes.unlink()
        m = {p: (float(np.mean(out[label][p])), float(np.std(out[label][p]))) for p in out[label]}
        out[label] = {"on_road_mean": float(np.mean(onroad)),
                      **{p: {"mean": m[p][0], "std": m[p][1]} for p in m}}
        lfu=m["LFU"][0]; fx=m["FixedW0.2"][0]; ad=m["AdaptiveTC"][0]
        print(f"[{label}] on-road~{np.mean(onroad):.0f}  LFU={lfu:.2f}  Fixed={fx:.2f}({fx-lfu:+.2f})  Adaptive={ad:.2f}({ad-lfu:+.2f})")
    out["_meta"] = {"seeds": SEEDS, "elapsed_s": time.time()-t0, "source": "SUMO 1.27 Krauss FCD"}
    (ROOT/"experiments"/"results"/"sumo_validation.json").write_text(json.dumps(out, indent=2))
    print(f"Total elapsed: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
