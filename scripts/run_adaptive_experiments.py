#!/usr/bin/env python
"""
Density-adaptive urgency weighting (#2) experiment.

Compares AdaptiveTC against fixed-W TrajectoryCache (W=0.2) and LFU across the
full density sweep, 5 seeds each, at alpha=0.8. The claim under test: adaptive-W
matches or beats fixed-W at every density and, crucially, avoids fixed-W's loss
at high density -- all without a manually tuned weight.
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from trajectorycache.cache import build_cache
from trajectorycache.utils.config import load_config
from trajectorycache.simulation.runner import SimulationRunner
from trajectorycache.evaluation.metrics import compute_metrics

SEEDS = [84810, 15592, 4278, 98196, 37048]
DENSITIES = [50, 100, 200, 400, 600]
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "simulation.yaml"

def multi(policy, n, kw):
    vals = []
    for s in SEEDS:
        cfg = load_config(CONFIG); cfg.seed = s; cfg.n_vehicles = n
        cache = build_cache(policy, cfg.cache_capacity, **kw)
        vals.append(compute_metrics(SimulationRunner(cache=cache, config=cfg).run()).miss_rate*100)
    return float(np.mean(vals)), float(np.std(vals))

def main():
    t0 = time.time()
    out = {"density_adaptive_sweep": {}}
    print(f"{'n':>4} | {'LFU':>14} | {'FixedW0.2':>14} | {'AdaptiveTC':>14} | fixedD | adaptD")
    print("-"*80)
    for n in DENSITIES:
        lfu_m, lfu_s = multi('lfu', n, {})
        fx_m, fx_s = multi('trajectory', n, {'urgency_weight':0.2})
        ad_m, ad_s = multi('adaptive', n, {'w_max':0.2})
        out["density_adaptive_sweep"][n] = {
            "LFU": {"mean": lfu_m, "std": lfu_s},
            "FixedW0.2": {"mean": fx_m, "std": fx_s, "margin_vs_lfu": fx_m-lfu_m},
            "AdaptiveTC": {"mean": ad_m, "std": ad_s, "margin_vs_lfu": ad_m-lfu_m},
        }
        print(f"{n:>4} | {lfu_m:6.2f}±{lfu_s:4.2f} | {fx_m:6.2f}±{fx_s:4.2f} | {ad_m:6.2f}±{ad_s:4.2f} | {fx_m-lfu_m:+5.2f} | {ad_m-lfu_m:+5.2f}")
    out["_meta"] = {"seeds": SEEDS, "densities": DENSITIES, "elapsed_s": time.time()-t0,
                    "adaptive_params": {"w_max":0.2, "rho_low":16.0, "rho_high":64.0}}
    p = Path(__file__).resolve().parents[1] / "experiments" / "results" / "adaptive_experiments.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {p}\nTotal elapsed: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
