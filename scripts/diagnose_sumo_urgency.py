"""
Diagnosis: WHY does TC lose to LFU under SUMO Krauss traffic?

Measures, on real SUMO FCD replays at three traffic levels:
  1. SATURATION: dispersion of raw urgency across catalog items
     (coefficient of variation; fraction of items with >0 urgency).
  2. PREDICTIVENESS: Spearman rank correlation between an item's current
     urgency (resp. sliding-window popularity) and its request count over
     the NEXT t_pred-second window. If urgency's rho ~ 0 while popularity's
     is high, the urgency term is noise and W>0 must hurt.
  3. NORMALIZATION EFFECT: spread of min-max-normalized urgency vs
     max-normalized urgency among cached-size subsets.
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
SCRATCH = Path(__file__).resolve().parents[1] / "sumo"

ROAD_LEN, ACTIVE_ZONE, N_ITEMS = 10000.0, 1600.0, 200
T_PRED, ALPHA_D, R_REL = 30.0, 0.1, 800.0
POP_WINDOW = 300.0
SEED = 84810


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
    steps = []
    for ts in ET.parse(fcd).getroot():
        vehs = []
        for v in ts:
            ang = float(v.get("angle"))
            direction = 1 if (ang < 180) else -1
            vehs.append({"x": float(v.get("x")), "speed": float(v.get("speed")),
                         "direction": direction})
        steps.append((float(ts.get("time")), vehs))
    return steps


def raw_urgency(item_loc, vehicles):
    total, contributors = 0.0, 0
    for veh in vehicles:
        x_v, speed, direction = veh["x"], veh["speed"], veh["direction"]
        if speed <= 0:
            continue
        x_hat = x_v + speed * direction * T_PRED
        if abs(x_hat - item_loc) > R_REL:
            continue
        tte = abs(item_loc - x_v) / speed
        total += 1.0 / (1.0 + ALPHA_D * tte)
        contributors += 1
    return total, contributors


def diagnose(label, vph):
    routes = SCRATCH / f"diag_routes_{label}.rou.xml"
    fcd = SCRATCH / f"diag_fcd_{label}.xml"
    write_routes(routes, vph, end=600)
    run_sumo(routes, fcd, end=600, seed=SEED)
    steps = parse_fcd(fcd)

    cat = ContentCatalog(n_items=N_ITEMS, road_length=ROAD_LEN,
                         active_zone_length=ACTIVE_ZONE, zipf_alpha=0.8, seed=SEED)
    loc_map = cat.location_map()
    item_ids = sorted(loc_map)
    locs = np.array([loc_map[i] for i in item_ids])

    # Pre-generate the full request stream once (same as validation script)
    start = next((i for i, (t, vs) in enumerate(steps) if len(vs) >= 20), 0)
    window = steps[start:start + 400]
    req_stream = []   # (t, item_id)
    for (t, vehs) in window:
        for item in cat.generate_vehicle_requests(vehicles=vehs, r_request=800.0):
            req_stream.append((t, item.item_id))

    reqs_by_t = defaultdict(list)
    for t, iid in req_stream:
        reqs_by_t[t].append(iid)

    # Sliding-window popularity counter
    req_times = defaultdict(deque)

    cv_list, frac_nonzero, contrib_mean = [], [], []
    rho_urg, rho_pop = [], []
    minmax_span_ratio = []   # (raw spread / raw mean) vs min-max always [0,1]
    on_road = []

    times = [t for (t, _) in window]
    for k, (t, vehs) in enumerate(window):
        # update popularity window
        for iid in reqs_by_t.get(t, []):
            req_times[iid].append(t)
        for dq in req_times.values():
            while dq and dq[0] < t - POP_WINDOW:
                dq.popleft()

        if k < 60 or k % 10 != 0:   # warmup, then sample every 10 steps
            continue
        on_road.append(len(vehs))

        u = np.zeros(len(item_ids))
        contribs = np.zeros(len(item_ids))
        for j, iid in enumerate(item_ids):
            u[j], contribs[j] = raw_urgency(locs[j], vehs)
        p = np.array([len(req_times[iid]) for iid in item_ids], dtype=float)

        # future requests within next T_PRED seconds
        fut = np.zeros(len(item_ids))
        idx = {iid: j for j, iid in enumerate(item_ids)}
        for t2 in range(int(t) + 1, int(t + T_PRED) + 1):
            for iid in reqs_by_t.get(float(t2), []):
                fut[idx[iid]] += 1

        nz = u > 0
        frac_nonzero.append(nz.mean())
        contrib_mean.append(contribs[nz].mean() if nz.any() else 0.0)
        if u.mean() > 0:
            cv_list.append(u.std() / u.mean())
            # min-max noise amplification: raw relative spread
            minmax_span_ratio.append((u.max() - u.min()) / (u.max() + 1e-9))
        if fut.sum() > 0:
            r_u = spearmanr(u, fut).statistic
            r_p = spearmanr(p, fut).statistic
            if not np.isnan(r_u):
                rho_urg.append(r_u)
            if not np.isnan(r_p):
                rho_pop.append(r_p)

    routes.unlink(missing_ok=True)
    fcd.unlink(missing_ok=True)

    res = {
        "vph": vph,
        "on_road_mean": float(np.mean(on_road)),
        "urgency_CV_mean": float(np.mean(cv_list)),
        "frac_items_with_nonzero_urgency": float(np.mean(frac_nonzero)),
        "mean_contributing_vehicles_per_item": float(np.mean(contrib_mean)),
        "raw_relative_span_(max-min)/max": float(np.mean(minmax_span_ratio)),
        "spearman_urgency_vs_future_requests": float(np.mean(rho_urg)),
        "spearman_popularity_vs_future_requests": float(np.mean(rho_pop)),
        "n_samples": len(rho_urg),
    }
    print(f"\n=== {label} (vph={vph}) ===")
    for k2, v2 in res.items():
        print(f"  {k2}: {v2:.4f}" if isinstance(v2, float) else f"  {k2}: {v2}")
    return res


if __name__ == "__main__":
    out = {}
    for label, vph in [("sparse", 150), ("moderate", 1100), ("high", 2600)]:
        out[label] = diagnose(label, vph)
    (SCRATCH / "diagnosis_results.json").write_text(json.dumps(out, indent=2))
    print("\nSaved to diagnosis_results.json")
