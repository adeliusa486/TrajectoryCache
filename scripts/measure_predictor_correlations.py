#!/usr/bin/env python
"""
Structured measurement of how well each candidate signal predicts near-term
demand under real SUMO Krauss traffic, for BOTH content geographies. Saves a
JSON keyed by geometry x traffic x predictor -> mean Spearman rho vs the
realized request count in the next T_PRED seconds. Feeds the diagnosis figure.

Predictors:
  popularity     : sliding-window request count (what LFU ranks by)
  urgency_TC     : the paper's x_hat-extrapolation urgency term
  exposure       : vehicles with the item in their forward request window now
                   (what EDC ranks on, multiplied by popularity)
"""
import sys, subprocess, json
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict, deque

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trajectorycache.content.catalog import ContentCatalog

SUMO_BIN = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe"
SUMO_DIR = ROOT / "sumo"
NET = SUMO_DIR / "highway.net.xml"

ROAD_LEN, N_ITEMS = 10000.0, 200
T_PRED, ALPHA_D, R_REL, R_REQ = 30.0, 0.1, 800.0, 800.0
POP_WINDOW = 300.0
SEEDS = [84810, 15592, 4278]


def write_routes(path, vph, end):
    path.write_text(f'''<routes>
    <vType id="car" carFollowModel="Krauss" maxSpeed="33.33" speedFactor="normc(1,0.1,0.7,1.3)" minGap="2.5" accel="2.6" decel="4.5" sigma="0.5"/>
    <route id="rAB" edges="AB"/>
    <route id="rBA" edges="BA"/>
    <flow id="fAB" type="car" route="rAB" begin="0" end="{end}" vehsPerHour="{vph}"/>
    <flow id="fBA" type="car" route="rBA" begin="0" end="{end}" vehsPerHour="{vph}"/>
</routes>''')


def get_steps(tag, vph, seed):
    routes = SUMO_DIR / f"pc_routes_{tag}.rou.xml"
    fcd = SUMO_DIR / f"pc_fcd_{tag}.xml"
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
    return steps


def urgency_tc(loc, vehs):
    tot = 0.0
    for v in vehs:
        if v["speed"] <= 0:
            continue
        x_hat = v["x"] + v["speed"] * v["direction"] * T_PRED
        if abs(x_hat - loc) > R_REL:
            continue
        tot += 1.0 / (1.0 + ALPHA_D * abs(loc - v["x"]) / v["speed"])
    return tot


def exposure(loc, vehs):
    tot = 0.0
    for v in vehs:
        if v["speed"] <= 0:
            continue
        d = (loc - v["x"]) * v["direction"]
        if 0 < d <= R_REQ:
            tot += 1.0
    return tot


def measure(zone, vph, seed, tag):
    steps = get_steps(tag, vph, seed)
    cat = ContentCatalog(n_items=N_ITEMS, road_length=ROAD_LEN,
                         active_zone_length=zone, zipf_alpha=0.8, seed=seed)
    loc_map = cat.location_map()
    ids = sorted(loc_map)
    locs = np.array([loc_map[i] for i in ids])
    idx = {iid: j for j, iid in enumerate(ids)}

    start = next((i for i, (t, vs) in enumerate(steps) if len(vs) >= 20), 0)
    window = steps[start:start + 400]
    reqs_by_t = defaultdict(list)
    for (t, vehs) in window:
        for item in cat.generate_vehicle_requests(vehicles=vehs, r_request=R_REQ):
            reqs_by_t[t].append(item.item_id)

    req_times = defaultdict(deque)
    rho = defaultdict(list)
    for k, (t, vehs) in enumerate(window):
        for iid in reqs_by_t.get(t, []):
            req_times[iid].append(t)
        for dq in req_times.values():
            while dq and dq[0] < t - POP_WINDOW:
                dq.popleft()
        if k < 60 or k % 10 != 0:
            continue
        pop = np.array([len(req_times[iid]) for iid in ids], float)
        u = np.array([urgency_tc(l, vehs) for l in locs])
        e = np.array([exposure(l, vehs) for l in locs])
        fut = np.zeros(len(ids))
        for t2 in range(int(t) + 1, int(t + T_PRED) + 1):
            for iid in reqs_by_t.get(float(t2), []):
                fut[idx[iid]] += 1
        if fut.sum() == 0:
            continue
        for name, arr in [("popularity", pop), ("urgency_TC", u), ("exposure", e),
                          ("pop_x_exposure", pop * e)]:
            r = spearmanr(arr, fut).statistic
            if not np.isnan(r):
                rho[name].append(r)
    return {name: float(np.mean(v)) for name, v in rho.items()}


def main():
    out = {}
    for geo_label, zone in [("hotspot1600", 1600.0), ("corridor10000", 10000.0)]:
        out[geo_label] = {}
        for tr_label, vph in [("sparse", 150), ("moderate", 1100), ("high", 2600)]:
            per_seed = [measure(zone, vph, s, f"{geo_label}_{tr_label}_{s}") for s in SEEDS]
            agg = {k: float(np.mean([d[k] for d in per_seed])) for k in per_seed[0]}
            out[geo_label][tr_label] = agg
            print(f"{geo_label:14s} {tr_label:9s} " +
                  " ".join(f"{k}={v:+.3f}" for k, v in agg.items()), flush=True)
    out["_meta"] = {"seeds": SEEDS, "t_pred_s": T_PRED,
                    "metric": "mean Spearman rho vs next-30s request count"}
    (ROOT / "experiments" / "results" / "predictor_correlations.json").write_text(
        json.dumps(out, indent=2))
    print("Saved predictor_correlations.json")


if __name__ == "__main__":
    main()
