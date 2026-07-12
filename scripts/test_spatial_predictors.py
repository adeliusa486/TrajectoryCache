"""
Predictor shoot-out on real SUMO FCD traffic.

Candidates for ranking items by next-T_PRED-window demand:
  A. popularity      : sliding-window request count (what LFU ranks by)
  B. urgency_old     : paper's x_hat extrapolation urgency (known ~0)
  C. exposure        : direction-aware count of vehicles with item in their
                       ACTUAL request window (0 < (loc-x)*dir <= 800)
  D. exposure_tte    : same, TTE-decay-weighted
  E. expected_demand : popularity * exposure  (multiplicative)
  F. expected_demand_tte : popularity * exposure_tte

Metric: mean Spearman rho vs realized request counts in next 30 s.
"""
import sys, subprocess
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
SCRATCH = Path(__file__).resolve().parents[1] / "sumo"

ROAD_LEN, ACTIVE_ZONE, N_ITEMS = 10000.0, 1600.0, 200
T_PRED, ALPHA_D, R_REL, R_REQ = 30.0, 0.1, 800.0, 800.0
POP_WINDOW = 300.0
SEED = 84810


def write_routes(path, vph, end):
    path.write_text(f'''<routes>
    <vType id="car" carFollowModel="Krauss" maxSpeed="33.33" speedFactor="normc(1,0.1,0.7,1.3)" minGap="2.5" accel="2.6" decel="4.5" sigma="0.5"/>
    <route id="rAB" edges="AB"/>
    <route id="rBA" edges="BA"/>
    <flow id="fAB" type="car" route="rAB" begin="0" end="{end}" vehsPerHour="{vph}"/>
    <flow id="fBA" type="car" route="rBA" begin="0" end="{end}" vehsPerHour="{vph}"/>
</routes>''')


def get_steps(label, vph):
    routes = SCRATCH / f"pred_routes_{label}.rou.xml"
    fcd = SCRATCH / f"pred_fcd_{label}.xml"
    write_routes(routes, vph, end=600)
    subprocess.run([SUMO_BIN, "-n", str(NET), "-r", str(routes),
                    "--fcd-output", str(fcd), "--step-length", "1",
                    "--begin", "0", "--end", "600", "--seed", str(SEED),
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


def urgency_old(loc, vehs):
    tot = 0.0
    for v in vehs:
        if v["speed"] <= 0:
            continue
        x_hat = v["x"] + v["speed"] * v["direction"] * T_PRED
        if abs(x_hat - loc) > R_REL:
            continue
        tot += 1.0 / (1.0 + ALPHA_D * abs(loc - v["x"]) / v["speed"])
    return tot


def exposure(loc, vehs, tte_weight=False):
    tot = 0.0
    for v in vehs:
        if v["speed"] <= 0:
            continue
        d = (loc - v["x"]) * v["direction"]
        if 0 < d <= R_REQ:
            tot += 1.0 / (1.0 + ALPHA_D * d / v["speed"]) if tte_weight else 1.0
    return tot


def evaluate(label, vph):
    steps = get_steps(label, vph)
    cat = ContentCatalog(n_items=N_ITEMS, road_length=ROAD_LEN,
                         active_zone_length=ACTIVE_ZONE, zipf_alpha=0.8, seed=SEED)
    loc_map = cat.location_map()
    item_ids = sorted(loc_map)
    locs = np.array([loc_map[i] for i in item_ids])
    idx = {iid: j for j, iid in enumerate(item_ids)}

    start = next((i for i, (t, vs) in enumerate(steps) if len(vs) >= 20), 0)
    window = steps[start:start + 400]

    reqs_by_t = defaultdict(list)
    for (t, vehs) in window:
        for item in cat.generate_vehicle_requests(vehicles=vehs, r_request=R_REQ):
            reqs_by_t[t].append(item.item_id)

    req_times = defaultdict(deque)
    rhos = defaultdict(list)
    on_road = []

    for k, (t, vehs) in enumerate(window):
        for iid in reqs_by_t.get(t, []):
            req_times[iid].append(t)
        for dq in req_times.values():
            while dq and dq[0] < t - POP_WINDOW:
                dq.popleft()
        if k < 60 or k % 10 != 0:
            continue
        on_road.append(len(vehs))

        pop = np.array([len(req_times[iid]) for iid in item_ids], dtype=float)
        u_old = np.array([urgency_old(l, vehs) for l in locs])
        expo = np.array([exposure(l, vehs) for l in locs])
        expo_t = np.array([exposure(l, vehs, tte_weight=True) for l in locs])

        fut = np.zeros(len(item_ids))
        for t2 in range(int(t) + 1, int(t + T_PRED) + 1):
            for iid in reqs_by_t.get(float(t2), []):
                fut[idx[iid]] += 1
        if fut.sum() == 0:
            continue

        cands = {
            "A_popularity": pop,
            "B_urgency_old": u_old,
            "C_exposure": expo,
            "D_exposure_tte": expo_t,
            "E_pop_x_exposure": pop * expo,
            "F_pop_x_exposure_tte": pop * expo_t,
        }
        for name, arr in cands.items():
            r = spearmanr(arr, fut).statistic
            if not np.isnan(r):
                rhos[name].append(r)

    print(f"\n=== {label} (vph={vph}, on-road~{np.mean(on_road):.0f}) ===")
    for name in sorted(rhos):
        print(f"  {name:24s} rho = {np.mean(rhos[name]):+.4f}  (n={len(rhos[name])})")


if __name__ == "__main__":
    for label, vph in [("sparse", 150), ("moderate", 1100), ("high", 2600)]:
        evaluate(label, vph)
