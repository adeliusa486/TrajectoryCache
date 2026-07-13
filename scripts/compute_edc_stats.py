#!/usr/bin/env python
"""
Rigorous statistics for EDC vs LFU across every evaluated condition.

For each (dataset, condition) with paired per-seed miss rates:
  - paired mean difference (EDC - LFU); negative = EDC better
  - two-sided Wilcoxon signed-rank p-value
  - matched-pairs rank-biserial effect size r = (W+ - W-)/(W+ + W-)
  - percentile bootstrap 95% CI on the mean paired difference (10k resamples)

Holm-Bonferroni correction is applied across the full family of SUMO
conditions (the primary confirmatory family). The synthetic-simulator
conditions are reported separately (exploratory / context).

Consumes:
  experiments/results/edc_sumo_10seed.json          (single-RSU grid)
  experiments/results/sumo_multirsu_edc_10seed.json (multi-RSU grid)
  experiments/results/synthetic_edc_10seed.json     (synthetic suite; optional)
Writes:
  experiments/results/edc_statistics.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
RNG = np.random.default_rng(20260712)


def rank_biserial(diffs: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation for a Wilcoxon signed-rank test."""
    d = diffs[diffs != 0]
    if d.size == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(d))) + 1
    w_plus = ranks[d > 0].sum()
    w_minus = ranks[d < 0].sum()
    total = w_plus + w_minus
    return float((w_plus - w_minus) / total) if total else 0.0


def bootstrap_ci(diffs: np.ndarray, n_boot: int = 10000, alpha: float = 0.05):
    means = np.array([RNG.choice(diffs, size=diffs.size, replace=True).mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def compare(edc: list[float], lfu: list[float]) -> dict:
    edc = np.asarray(edc, float)
    lfu = np.asarray(lfu, float)
    diffs = edc - lfu  # negative = EDC better
    # Wilcoxon needs at least one non-zero difference
    if np.allclose(diffs, 0):
        p = 1.0
    else:
        p = float(wilcoxon(edc, lfu, alternative="two-sided",
                           zero_method="wilcox").pvalue)
    lo, hi = bootstrap_ci(diffs)
    return {
        "n": int(diffs.size),
        "edc_mean": float(edc.mean()),
        "lfu_mean": float(lfu.mean()),
        "mean_diff": float(diffs.mean()),
        "ci95_diff": [round(lo, 3), round(hi, 3)],
        "wilcoxon_p_two_sided": p,
        "rank_biserial_r": round(rank_biserial(diffs), 3),
        "edc_better": bool(diffs.mean() < 0),
    }


def holm(pairs: list[tuple[str, float]]) -> dict[str, float]:
    """Holm-Bonferroni step-down. pairs = [(label, p), ...]."""
    m = len(pairs)
    order = sorted(range(m), key=lambda i: pairs[i][1])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pairs[idx][1]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return {pairs[i][0]: round(adj[i], 4) for i in range(m)}


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def main():
    out = {"sumo_confirmatory_family": {}, "synthetic_context": {}, "_meta": {}}

    # ---- SUMO confirmatory family (single-RSU + multi-RSU) ----
    family_p = []
    single = load("edc_sumo_10seed.json")
    if single:
        for geo in ["hotspot1600", "corridor10000"]:
            for tr in ["sparse", "moderate", "high"]:
                cell = single[geo][tr]
                res = compare(cell["EDC"]["per_seed"], cell["LFU"]["per_seed"])
                label = f"single/{geo}/{tr}"
                out["sumo_confirmatory_family"][label] = res
                family_p.append((label, res["wilcoxon_p_two_sided"]))

    multi = load("sumo_multirsu_edc_10seed.json")
    if multi:
        for geo in ["hotspot1600", "corridor10000"]:
            for tr in ["sparse", "moderate", "high"]:
                cell = multi[geo][tr]
                res = compare(cell["EDC"]["per_seed"], cell["LFU"]["per_seed"])
                label = f"multi/{geo}/{tr}"
                out["sumo_confirmatory_family"][label] = res
                family_p.append((label, res["wilcoxon_p_two_sided"]))

    holm_adj = holm(family_p)
    for label, padj in holm_adj.items():
        out["sumo_confirmatory_family"][label]["holm_adjusted_p"] = padj
        out["sumo_confirmatory_family"][label]["significant_holm_0.05"] = bool(padj < 0.05)

    # ---- Synthetic context (reported separately, not Holm-corrected here) ----
    syn = load("synthetic_edc_10seed.json")
    if syn:
        for alpha in ["alpha0.8", "alpha0.5"]:
            for cond in ["platoon", "independent"]:
                cell = syn[alpha][cond]
                if "EDC" in cell and "LFU" in cell:
                    res = compare(cell["EDC"]["per_seed"], cell["LFU"]["per_seed"])
                    out["synthetic_context"][f"{alpha}/{cond}"] = res
        if "density_sweep_alpha0.8" in syn:
            for n, cell in syn["density_sweep_alpha0.8"].items():
                res = compare(cell["EDC"]["per_seed"], cell["LFU"]["per_seed"])
                out["synthetic_context"][f"density/n={n}"] = res

    out["_meta"] = {
        "test": "two-sided Wilcoxon signed-rank, EDC vs LFU, paired by seed",
        "correction": "Holm-Bonferroni across SUMO confirmatory family",
        "effect_size": "matched-pairs rank-biserial correlation",
        "ci": "percentile bootstrap 95% (10000 resamples), seed 20260712",
        "family_size": len(family_p),
    }
    (RES / "edc_statistics.json").write_text(json.dumps(out, indent=2))

    # ---- Print ----
    print("==== SUMO CONFIRMATORY FAMILY (Holm-corrected) ====")
    print(f"{'condition':30s} {'EDC':>6s} {'LFU':>6s} {'diff':>6s} {'CI95':>16s} "
          f"{'p':>7s} {'p_holm':>7s} {'r':>6s} sig")
    for label, r in out["sumo_confirmatory_family"].items():
        ci = f"[{r['ci95_diff'][0]:+.2f},{r['ci95_diff'][1]:+.2f}]"
        sig = "*" if r["significant_holm_0.05"] else ""
        print(f"{label:30s} {r['edc_mean']:6.2f} {r['lfu_mean']:6.2f} {r['mean_diff']:+6.2f} "
              f"{ci:>16s} {r['wilcoxon_p_two_sided']:7.4f} {r['holm_adjusted_p']:7.4f} "
              f"{r['rank_biserial_r']:+6.2f} {sig}")

    if out["synthetic_context"]:
        print("\n==== SYNTHETIC CONTEXT (uncorrected) ====")
        for label, r in out["synthetic_context"].items():
            ci = f"[{r['ci95_diff'][0]:+.2f},{r['ci95_diff'][1]:+.2f}]"
            print(f"{label:30s} {r['edc_mean']:6.2f} {r['lfu_mean']:6.2f} {r['mean_diff']:+6.2f} "
                  f"{ci:>16s} p={r['wilcoxon_p_two_sided']:.4f} r={r['rank_biserial_r']:+.2f}")

    print(f"\nSaved to {RES / 'edc_statistics.json'}")


if __name__ == "__main__":
    main()
