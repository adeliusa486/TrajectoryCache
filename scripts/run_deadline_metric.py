#!/usr/bin/env python
"""
Deadline-aware metric test: does TC concentrate its misses on LESS time-critical
requests compared to LFU, even when plain miss-rate ties or loses?

For each request we log:
  - hit/miss
  - TTE_at_request: time until the requesting vehicle reaches the item's exact
    location, given its current speed (distance / speed). Lower = more urgent.

We then compute:
  - plain miss rate (sanity check vs prior results)
  - urgency-weighted miss cost = mean over ALL requests of [miss_indicator / (eps+TTE)]
    i.e. expected "urgency wasted on misses" per request. Lower is better.
  - mean TTE of misses only (do misses skew toward high-TTE = less urgent?)
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from trajectorycache.cache import build_cache
from trajectorycache.utils.config import load_config
from trajectorycache.simulation.highway import HighwaySimulation
from trajectorycache.content.catalog import ContentCatalog

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [84810, 15592, 4278, 98196, 37048]
EPS = 0.5

def run_logged(policy, kw, n_vehicles, seed):
    cfg = load_config(ROOT / "configs" / "simulation.yaml")
    cfg.seed = seed
    cfg.n_vehicles = n_vehicles
    hw = HighwaySimulation(road_length=cfg.road_length, n_vehicles=cfg.n_vehicles, dt=cfg.dt,
                           mean_speed=cfg.mean_speed, speed_std=cfg.speed_std,
                           platoon_size=cfg.platoon_size, platoon_gap=cfg.platoon_gap, seed=seed)
    cat = ContentCatalog(n_items=cfg.n_items, road_length=cfg.road_length,
                         active_zone_length=cfg.active_zone_length, zipf_alpha=cfg.zipf_alpha, seed=seed)
    loc_map = cat.location_map()
    cache = build_cache(policy, cfg.cache_capacity, **kw)
    cache.clear()

    miss_flags, ttes = [], []
    total_steps = cfg.warmup_steps + cfg.n_steps
    for step in range(total_steps):
        vehicles = hw.step()
        t = hw.current_time
        if step == cfg.warmup_steps:
            cache.reset_stats(); miss_flags.clear(); ttes.clear()
        reqs_with_veh = cat.generate_vehicle_requests_with_source(vehicles=vehicles, r_request=cfg.r_rel) \
            if hasattr(cat, "generate_vehicle_requests_with_source") else None
        if reqs_with_veh is None:
            # Fallback: recompute requester association ourselves (mirrors catalog logic)
            for veh in vehicles:
                x_v, direction, speed = veh["x"], veh.get("direction",1), veh.get("speed",0.0)
                if speed <= 0:
                    continue
                cands = []
                for iid, item in cat._items.items():
                    dist = (item.location - x_v) * direction
                    if 0 < dist <= cfg.r_rel:
                        cands.append((iid, dist))
                if not cands:
                    continue
                ids = [c[0] for c in cands]
                w = cat._popularity_weights[ids]
                if w.sum() == 0:
                    continue
                w = w / w.sum()
                idx = int(cat._rng.choice(len(ids), p=w))
                iid, dist = cands[idx]
                item = cat._items[iid]
                hit = cache.request(item_id=item.item_id, item_location=item.location,
                                    current_time=t, vehicles=vehicles, catalog=loc_map)
                tte = dist / speed
                miss_flags.append(0 if hit else 1)
                ttes.append(tte)
    miss_flags = np.array(miss_flags); ttes = np.array(ttes)
    plain_miss = miss_flags.mean() * 100
    weighted_cost = float(np.mean(miss_flags / (EPS + ttes)))
    mean_tte_miss = float(ttes[miss_flags==1].mean()) if miss_flags.sum() else 0.0
    mean_tte_hit = float(ttes[miss_flags==0].mean()) if (1-miss_flags).sum() else 0.0
    return plain_miss, weighted_cost, mean_tte_miss, mean_tte_hit

def main():
    t0 = time.time()
    out = {}
    for n in [200, 400, 600]:
        out[str(n)] = {}
        for label, policy, kw in [("LFU","lfu",{}), ("FixedW0.2","trajectory",{"urgency_weight":0.2}),
                                   ("AdaptiveTC","adaptive",{"w_max":0.2})]:
            pm, wc, mtm, mth = [], [], [], []
            for seed in SEEDS:
                p, w, mm, mh = run_logged(policy, kw, n, seed)
                pm.append(p); wc.append(w); mtm.append(mm); mth.append(mh)
            out[str(n)][label] = {
                "plain_miss_pct": [float(np.mean(pm)), float(np.std(pm))],
                "weighted_miss_cost": [float(np.mean(wc)), float(np.std(wc))],
                "mean_tte_of_misses_s": float(np.mean(mtm)),
                "mean_tte_of_hits_s": float(np.mean(mth)),
            }
            print(f"n={n} {label:12s} miss={np.mean(pm):.2f}%  weighted_cost={np.mean(wc):.5f}  "
                  f"TTE(miss)={np.mean(mtm):.2f}s  TTE(hit)={np.mean(mth):.2f}s")
    out["_meta"] = {"seeds": SEEDS, "eps": EPS, "elapsed_s": time.time()-t0}
    (ROOT/"experiments"/"results"/"deadline_metric.json").write_text(json.dumps(out, indent=2))
    print(f"Total elapsed: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
