"""
N6: signal-quality diagnostic on the REAL NGSIM tier.

Measures, on real i-80 trajectories, the Spearman rank correlation between each
scoring signal and the number of requests each item actually receives over the
next T_PRED seconds:
  - urgency_SU : the spatial-urgency (lookahead extrapolation) signal
  - popularity : sliding-window request count
  - exposure   : vehicles with the item in their forward request window now

This verifies (or corrects) the abstract's "rho ~ 0.05 on real trajectories"
claim. Uses the SAME 535 m scenario as run_real_ngsim.py.
"""
from __future__ import annotations

import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from adapters.ngsim_adapter import load_ngsim_steps
from trajectorycache.content.catalog import ContentCatalog

NGSIM_CSV = ROOT / "data" / "raw" / "ngsim" / "ngsim_i80_win1.csv"
N_ITEMS, R_REQ, T_PRED, ALPHA_D = 200, 150.0, 30.0, 0.1
POP_WINDOW = 300.0
SEEDS = [84810, 15592, 4278, 98196, 37048]
WARMUP, MEASURE = 150, 600


def urgency_su(loc, vehs):
    tot = 0.0
    for v in vehs:
        s = v["speed"]
        if s <= 0:
            continue
        x_hat = v["x"] + s * v["direction"] * T_PRED
        if abs(x_hat - loc) > R_REQ:
            continue
        tot += 1.0 / (1.0 + ALPHA_D * abs(loc - v["x"]) / s)
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


def main():
    steps, meta = load_ngsim_steps(NGSIM_CSV)
    window = steps[: WARMUP + MEASURE]
    print(f"NGSIM i-80 seg={meta['segment_length_m']:.0f}m, mean on-road "
          f"{meta['mean_on_road']:.0f}, {meta['mean_speed_mps']*3.6:.0f} km/h", flush=True)

    agg = defaultdict(list)
    for seed in SEEDS:
        cat = ContentCatalog(n_items=N_ITEMS, road_length=meta["segment_length_m"],
                             active_zone_length=meta["segment_length_m"], zipf_alpha=0.8, seed=seed)
        loc_map = cat.location_map()
        ids = sorted(loc_map)
        locs = np.array([loc_map[i] for i in ids])
        idx = {iid: j for j, iid in enumerate(ids)}

        reqs_by_t = defaultdict(list)
        for (t, vehs) in window:
            for item in cat.generate_vehicle_requests(vehicles=vehs, r_request=R_REQ):
                reqs_by_t[int(t)].append(item.item_id)

        req_times = defaultdict(deque)
        rho = defaultdict(list)
        for k, (t, vehs) in enumerate(window):
            ti = int(t)
            for iid in reqs_by_t.get(ti, []):
                req_times[iid].append(ti)
            for dq in req_times.values():
                while dq and dq[0] < ti - POP_WINDOW:
                    dq.popleft()
            if k < 60 or k % 10 != 0:
                continue
            pop = np.array([len(req_times[iid]) for iid in ids], float)
            u = np.array([urgency_su(l, vehs) for l in locs])
            e = np.array([exposure(l, vehs) for l in locs])
            fut = np.zeros(len(ids))
            for t2 in range(ti + 1, ti + int(T_PRED) + 1):
                for iid in reqs_by_t.get(t2, []):
                    fut[idx[iid]] += 1
            if fut.sum() == 0:
                continue
            for name, arr in [("urgency_SU", u), ("popularity", pop),
                              ("exposure", e), ("pop_x_exposure", pop * e)]:
                r = spearmanr(arr, fut).statistic
                if not np.isnan(r):
                    rho[name].append(r)
        for name, v in rho.items():
            agg[name].append(float(np.mean(v)))

    print("\n=== NGSIM real-tier signal correlation (mean Spearman rho vs next-30s demand) ===")
    out = {}
    for name in ["urgency_SU", "popularity", "exposure", "pop_x_exposure"]:
        m, s = float(np.mean(agg[name])), float(np.std(agg[name]))
        out[name] = (m, s)
        print(f"  {name:16s} rho = {m:+.3f} +/- {s:.3f}  (over {len(agg[name])} seeds)")
    import json
    (ROOT / "experiments" / "results" / "ngsim_signal_correlation.json").write_text(
        json.dumps({k: {"mean": v[0], "std": v[1]} for k, v in out.items()}
                   | {"_meta": {"seeds": SEEDS, "segment_m": meta["segment_length_m"],
                                "mean_on_road": meta["mean_on_road"]}}, indent=2))
    print("saved ngsim_signal_correlation.json")


if __name__ == "__main__":
    main()
