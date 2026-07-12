"""
End-to-end SUMO validation of ExpectedDemandCache (EDC) vs LFU vs old TC.

Grid: {concentrated 1600m, dispersed 10000m} content geography
    x {sparse 150, moderate 1100, high 2600} veh/h per direction
    x 5 seeds. All policies replay the IDENTICAL request stream per run.
"""
import sys, subprocess, time, json
import xml.etree.ElementTree as ET
from pathlib import Path
from multiprocessing import Pool

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SUMO_BIN = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe"
NET = ROOT / "sumo" / "highway.net.xml"
SCRATCH = Path(__file__).resolve().parents[1] / "sumo"

ROAD_LEN, N_ITEMS, CAP = 10000.0, 200, 20
WARMUP, MEASURE = 100, 300
SEEDS = [84810, 15592, 4278, 98196, 37048]
POLICIES = [
    ("LFU", "lfu", {}),
    ("TC_W0.2", "trajectory", {"urgency_weight": 0.2}),
    ("EDC", "expected_demand", {}),
    ("EDC_tte", "expected_demand", {"tte_weight": True}),
]


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
    routes = SCRATCH / f"edc_routes_{tag}.rou.xml"
    fcd = SCRATCH / f"edc_fcd_{tag}.xml"
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

    # Pre-generate the request stream ONCE so every policy sees identical demand
    stream = []
    for (t, vehs) in window:
        reqs = cat.generate_vehicle_requests(vehicles=vehs, r_request=800.0)
        stream.append((t, vehs, [(it.item_id, it.location) for it in reqs]))

    out = {}
    for name, key, kw in POLICIES:
        cache = build_cache(key, CAP, **kw)
        cache.clear()
        for i, (t, vehs, reqs) in enumerate(stream):
            if i == WARMUP:
                cache.reset_stats()
            for iid, loc in reqs:
                cache.request(item_id=iid, item_location=loc, current_time=t,
                              vehicles=vehs, catalog=loc_map)
        out[name] = cache.summary()["miss_rate"]
    on_road = float(np.mean([len(vs) for _, vs, _ in stream]))
    return geo_label, traffic_label, seed, on_road, out


def main():
    t0 = time.time()
    jobs = []
    for geo_label, zone in [("hotspot1600", 1600.0), ("corridor10000", 10000.0)]:
        for traffic_label, vph in [("sparse", 150), ("moderate", 1100), ("high", 2600)]:
            for seed in SEEDS:
                jobs.append((geo_label, zone, traffic_label, vph, seed))

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
    summary["_meta"] = {"seeds": SEEDS, "elapsed_s": time.time() - t0,
                        "source": "SUMO 1.27 Krauss FCD",
                        "policies": [p[0] for p in POLICIES]}
    out_path = SCRATCH / "edc_sumo_results.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print("\n==== SUMMARY (miss rate %, mean over 5 seeds) ====")
    for geo in ["hotspot1600", "corridor10000"]:
        for tr in ["sparse", "moderate", "high"]:
            s = summary[geo][tr]
            line = f"{geo:14s} {tr:9s} onroad~{s['on_road_mean']:.0f}  "
            lfu = s["LFU"]["mean"]
            for p in ["LFU", "TC_W0.2", "EDC", "EDC_tte"]:
                line += f"{p}={s[p]['mean']:.2f}({s[p]['mean']-lfu:+.2f}) "
            print(line)
    print(f"\nElapsed {time.time()-t0:.0f}s. Saved to {out_path}")


if __name__ == "__main__":
    main()
