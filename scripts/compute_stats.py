#!/usr/bin/env python
"""
scripts/compute_stats.py
Compute Wilcoxon signed-rank test between TC and LFU per-seed miss rates
from SUMO (platooning) condition, and print ablation table from wsweep data.

Loads from the canonical JSON structure produced by run_multiseed.py:
  {sumo: {PolicyName: {miss_rate_mean, miss_rate_std, per_seed: [...]}, ...},
   simpy: {...}, seeds: [...]}
"""

import json
import sys
from pathlib import Path
from scipy import stats
import numpy as np

import argparse

# ---- Parse arguments ----
parser = argparse.ArgumentParser()
parser.add_argument("--input", default="experiments/results/alpha08/multiseed_alpha0.8.json")
parser.add_argument("--wsweep-input", default="experiments/results/wsweep/wsweep_results.json")
args = parser.parse_args()

alpha08_path = Path(args.input)
wsweep_path = Path(args.wsweep_input)

try:
    alpha08 = json.loads(alpha08_path.read_text())
except FileNotFoundError:
    print(f"Error: Could not find {alpha08_path}. Did you run scripts/run_multiseed.py?")
    sys.exit(1)

try:
    wsweep = json.loads(wsweep_path.read_text())
except FileNotFoundError:
    print(f"Warning: Could not find {wsweep_path}. Ablation stats will be skipped.")
    wsweep = None

# Validate JSON structure
assert (
    "sumo" in alpha08
), f"Expected 'sumo' key in {alpha08_path}; got keys: {list(alpha08.keys())}"
assert "simpy" in alpha08, f"Expected 'simpy' key in {alpha08_path}"
assert (
    "per_seed" in alpha08["sumo"]["TrajectoryCache"]
), "Missing per_seed array in sumo.TrajectoryCache — re-run scripts/run_multiseed.py to regenerate"

# Use SUMO (platooning) per-seed data for the significance test
# This is the primary evaluation condition reported in Table 1
tc_seeds_sumo = alpha08["sumo"]["TrajectoryCache"]["per_seed"]
lfu_seeds_sumo = alpha08["sumo"]["LFU"]["per_seed"]

assert (
    len(tc_seeds_sumo) == len(lfu_seeds_sumo) == 10
), f"Expected 10 seeds, got TC={len(tc_seeds_sumo)}, LFU={len(lfu_seeds_sumo)}"

# ---- Wilcoxon signed-rank test (SUMO condition) ----
stat, p = stats.wilcoxon(tc_seeds_sumo, lfu_seeds_sumo, alternative="less")
print(f"\n=== Wilcoxon signed-rank test: TC < LFU — SUMO (platooning), alpha=0.8 ===")
print(f"  n = {len(tc_seeds_sumo)}")
print(f"  TC  per-seed: {[round(v,2) for v in tc_seeds_sumo]}")
print(f"  LFU per-seed: {[round(v,2) for v in lfu_seeds_sumo]}")
print(f"  TC  mean={np.mean(tc_seeds_sumo):.4f}%  std={np.std(tc_seeds_sumo):.4f}%")
print(f"  LFU mean={np.mean(lfu_seeds_sumo):.4f}%  std={np.std(lfu_seeds_sumo):.4f}%")
print(
    f"  TC wins (lower miss rate) in {sum(t < l for t, l in zip(tc_seeds_sumo, lfu_seeds_sumo))}/10 seeds"
)
print(f"  statistic = {stat:.3f}, p-value = {p:.4f}")
if p < 0.05:
    print(
        "  => SIGNIFICANT (p < 0.05): TC has significantly lower miss rate than LFU under platooning"
    )
else:
    print("  => Not significant at p=0.05")

# ---- SimPy condition (independent traffic) ----
tc_seeds_simpy = alpha08["simpy"]["TrajectoryCache"]["per_seed"]
lfu_seeds_simpy = alpha08["simpy"]["LFU"]["per_seed"]
stat2, p2 = stats.wilcoxon(tc_seeds_simpy, lfu_seeds_simpy, alternative="less")
print(f"\n=== Wilcoxon signed-rank test: TC < LFU — SimPy (independent), alpha=0.8 ===")
print(f"  TC  mean={np.mean(tc_seeds_simpy):.4f}%  std={np.std(tc_seeds_simpy):.4f}%")
print(f"  LFU mean={np.mean(lfu_seeds_simpy):.4f}%  std={np.std(lfu_seeds_simpy):.4f}%")
print(f"  statistic = {stat2:.3f}, p-value = {p2:.4f}")
if p2 < 0.05:
    print("  => SIGNIFICANT: TC also beats LFU under independent traffic")
else:
    print("  => NOT significant: TC does NOT beat LFU under independent traffic (expected)")

# ---- Ablation table from wsweep ----
if wsweep is not None:
    print(f"\n=== Ablation: W sweep (LFU mean={wsweep['lfu_mean']:.2f}%) ===")
    print(f"{'W':>6}  {'TC mean%':>10}  {'TC std%':>8}  {'vs LFU':>8}")
    print("-" * 42)
    for w_str, vals in wsweep["w_sweep"].items():
        margin = vals["mean"] - wsweep["lfu_mean"]
        print(
            f"  {float(w_str):.1f}    {vals['mean']:>8.2f}%    {vals['std']:>6.2f}%   {margin:>+7.2f}%"
        )
