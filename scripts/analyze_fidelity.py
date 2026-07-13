#!/usr/bin/env python
"""
Fidelity-ladder analysis: quantify how the mobility model distorts caching
evaluation, at the MATCHED 535 m unidirectional scenario across three tiers:
  synthetic platoon  ->  SUMO Krauss  ->  real NGSIM i-80.

Outputs (experiments/results/fidelity_analysis.json + console):
  1. Per-tier policy ranking (by mean miss rate).
  2. Ranking (dis)agreement between tiers: Kendall tau_b and Spearman rho over
     the shared policy set --- the headline "does the model change the winner?"
     number.
  3. Per-policy gain vs LFU per tier, and the synthetic->real inflation factor
     for the spatial policies (TC, EDC).
  4. Sign-flip table: policies whose gain-vs-LFU changes sign between tiers.

Reads:
  matched_tiers_535m.json  (synthetic + sumo tiers)
  real_ngsim_i80.json      (real tier)
All three share scenario params by construction (see the two runner scripts).
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

RES = Path(__file__).resolve().parents[1] / "experiments" / "results"

# Policies present in all tiers (order is display order).
POLICIES = ["LRU", "FIFO", "Random", "LFU", "Proximity", "TC_W0.2", "EDC", "QLearning"]


def load_tiers():
    matched = json.loads((RES / "matched_tiers_535m.json").read_text())
    real = json.loads((RES / "real_ngsim_i80.json").read_text())
    tiers = {
        "synthetic": matched["synthetic"],
        "sumo": matched["sumo"],
        "real": real["policies"],
    }
    return tiers


def mean_of(tier, p):
    return tier[p]["mean"]


def ranking(tier, policies):
    """Return policies ordered best (lowest miss) -> worst."""
    return sorted(policies, key=lambda p: mean_of(tier, p))


def main():
    tiers = load_tiers()
    present = [p for p in POLICIES if all(p in tiers[t] for t in tiers)]

    out = {"tiers_order": list(tiers), "policies": present,
           "per_tier_mean": {}, "per_tier_rank": {}, "ranking_agreement": {},
           "gain_vs_lfu": {}, "inflation": {}, "sign_flips": []}

    # 1. means + rankings
    for t, tier in tiers.items():
        out["per_tier_mean"][t] = {p: round(mean_of(tier, p), 3) for p in present}
        rk = ranking(tier, present)
        out["per_tier_rank"][t] = {p: rk.index(p) + 1 for p in present}

    # 2. ranking agreement between tier pairs
    rank_vecs = {t: [out["per_tier_rank"][t][p] for p in present] for t in tiers}
    for a, b in combinations(tiers, 2):
        tau = kendalltau(rank_vecs[a], rank_vecs[b]).statistic
        rho = spearmanr(rank_vecs[a], rank_vecs[b]).statistic
        out["ranking_agreement"][f"{a}_vs_{b}"] = {
            "kendall_tau_b": round(float(tau), 3), "spearman_rho": round(float(rho), 3)}

    # 3. gain vs LFU per tier
    for t, tier in tiers.items():
        lfu = mean_of(tier, "LFU")
        out["gain_vs_lfu"][t] = {p: round(mean_of(tier, p) - lfu, 3) for p in present}

    # inflation: synthetic gain / real gain for spatial policies (guard tiny denom)
    for p in ["TC_W0.2", "EDC", "Proximity"]:
        if p not in present:
            continue
        gs = out["gain_vs_lfu"]["synthetic"][p]
        gr = out["gain_vs_lfu"]["real"][p]
        out["inflation"][p] = {
            "synthetic_gain": gs, "sumo_gain": out["gain_vs_lfu"]["sumo"][p],
            "real_gain": gr,
            "synth_over_real": round(gs / gr, 2) if abs(gr) > 1e-9 else None}

    # 4. sign flips of gain-vs-LFU across tiers
    for p in present:
        if p == "LFU":
            continue
        signs = {t: np.sign(out["gain_vs_lfu"][t][p]) for t in tiers}
        if len(set(s for s in signs.values() if s != 0)) > 1:
            out["sign_flips"].append({"policy": p, "gains": {t: out["gain_vs_lfu"][t][p] for t in tiers}})

    (RES / "fidelity_analysis.json").write_text(json.dumps(out, indent=2))

    # ---- print ----
    print("=== PER-TIER MISS RATE (%) and RANK ===")
    hdr = f"{'policy':10s}" + "".join(f"{t:>22s}" for t in tiers)
    print(hdr)
    for p in present:
        row = f"{p:10s}"
        for t in tiers:
            row += f"{out['per_tier_mean'][t][p]:>14.2f} (#{out['per_tier_rank'][t][p]})"
        print(row)

    print("\n=== RANKING AGREEMENT BETWEEN TIERS (lower = model changes the ranking) ===")
    for k, v in out["ranking_agreement"].items():
        print(f"  {k:24s} Kendall tau_b={v['kendall_tau_b']:+.2f}  Spearman rho={v['spearman_rho']:+.2f}")

    print("\n=== GAIN vs LFU (pp; negative = beats LFU) ===")
    print(f"{'policy':10s}" + "".join(f"{t:>12s}" for t in tiers))
    for p in present:
        if p == "LFU":
            continue
        print(f"{p:10s}" + "".join(f"{out['gain_vs_lfu'][t][p]:>12.2f}" for t in tiers))

    print("\n=== SPATIAL-GAIN INFLATION (synthetic vs real) ===")
    for p, v in out["inflation"].items():
        print(f"  {p:10s} synth={v['synthetic_gain']:+.2f}  sumo={v['sumo_gain']:+.2f}  "
              f"real={v['real_gain']:+.2f}  synth/real={v['synth_over_real']}")

    print("\n=== SIGN FLIPS (policy beats LFU in one tier, loses in another) ===")
    for f in out["sign_flips"]:
        print(f"  {f['policy']:10s} " + "  ".join(f"{t}={g:+.2f}" for t, g in f["gains"].items()))

    print(f"\nSaved fidelity_analysis.json")


if __name__ == "__main__":
    main()
