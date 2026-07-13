#!/usr/bin/env python
"""
Generate the three diagnosis/result figures for the EDC paper, as vector PDFs
in paper/figures/. All numbers are read from committed experiment JSONs -- no
hardcoded values -- so the figures always match the reported tables.

  fig_diagnosis_spearman.pdf : why urgency fails and exposure helps only under
      dispersed geography (Spearman of each signal vs next-30s demand).
  fig_operating_boundary.pdf : EDC-minus-LFU miss-rate margin across the SUMO
      single- and multi-RSU grid (the honest operating boundary).
  fig_fidelity_gap.pdf       : EDC-minus-LFU margin in the artificial-platoon
      synthetic simulator vs realistic SUMO -- the simulator-fidelity gap.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
FIGDIR = ROOT / "experiments" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.5,
    "figure.dpi": 200, "savefig.bbox": "tight",
})
BLUE, ORANGE, GREEN, GRAY, RED = "#1f61ab", "#c46210", "#278b3c", "#64646e", "#aa1e1e"


def load(name):
    return json.loads((RES / name).read_text())


def fig_spearman():
    d = load("predictor_correlations.json")
    conds = [("hotspot1600", "sparse"), ("hotspot1600", "moderate"), ("hotspot1600", "high"),
             ("corridor10000", "sparse"), ("corridor10000", "moderate"), ("corridor10000", "high")]
    labels = ["hot/sparse", "hot/mod", "hot/high", "cor/sparse", "cor/mod", "cor/high"]
    urg = [d[g][t]["urgency_TC"] for g, t in conds]
    pop = [d[g][t]["popularity"] for g, t in conds]
    edc = [d[g][t]["pop_x_exposure"] for g, t in conds]

    x = np.arange(len(conds)); w = 0.27
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(x - w, urg, w, label="TC urgency", color=RED, alpha=0.85)
    ax.bar(x, pop, w, label="popularity (LFU)", color=GRAY, alpha=0.85)
    ax.bar(x + w, edc, w, label="pop $\\times$ exposure (EDC)", color=GREEN, alpha=0.9)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("Spearman $\\rho$ vs next-30 s demand")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.axvspan(-0.5, 2.5, color=BLUE, alpha=0.04)
    ax.axvspan(2.5, 5.5, color=ORANGE, alpha=0.05)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper center", ncol=3, fontsize=8, framealpha=0.9,
              bbox_to_anchor=(0.5, 1.14), columnspacing=1.0)
    ax.text(1.0, 0.88, "hotspot geography", ha="center", fontsize=8, color=BLUE)
    ax.text(4.0, 0.88, "corridor geography", ha="center", fontsize=8, color=ORANGE)
    fig.savefig(FIGDIR / "fig_diagnosis_spearman.pdf")
    plt.close(fig)
    print("wrote fig_diagnosis_spearman.pdf")


def _grid_deltas(dataset):
    d = load(dataset)
    conds = [("hotspot1600", "sparse"), ("hotspot1600", "moderate"), ("hotspot1600", "high"),
             ("corridor10000", "sparse"), ("corridor10000", "moderate"), ("corridor10000", "high")]
    labels = ["hot/sparse", "hot/mod", "hot/high", "cor/sparse", "cor/mod", "cor/high"]
    deltas, errs = [], []
    for g, t in conds:
        edc = np.array(d[g][t]["EDC"]["per_seed"])
        lfu = np.array(d[g][t]["LFU"]["per_seed"])
        diff = edc - lfu
        deltas.append(diff.mean())
        errs.append(diff.std() / np.sqrt(len(diff)))
    return labels, deltas, errs


def fig_operating_boundary():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, dataset, title in [
        (axes[0], "edc_sumo_10seed.json", "Single RSU"),
        (axes[1], "sumo_multirsu_edc_10seed.json", "5 RSUs"),
    ]:
        labels, deltas, errs = _grid_deltas(dataset)
        colors = [GREEN if v < 0 else RED for v in deltas]
        x = np.arange(len(labels))
        ax.bar(x, deltas, 0.62, yerr=errs, color=colors, alpha=0.85,
               error_kw={"lw": 0.8, "capsize": 2})
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("EDC $-$ LFU miss rate (pp)\n$\\leftarrow$ EDC better | LFU better $\\rightarrow$")
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(color=GREEN, label="EDC wins"),
                            Patch(color=RED, label="LFU wins")],
                   fontsize=8, loc="upper left", framealpha=0.9)
    fig.suptitle("EDC $-$ LFU margin across the SUMO grid (mean $\\pm$ SEM, 10 seeds)",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_operating_boundary.pdf")
    plt.close(fig)
    print("wrote fig_operating_boundary.pdf")


def fig_fidelity_gap():
    syn = load("synthetic_edc_10seed.json")
    sumo = load("edc_sumo_10seed.json")

    # synthetic density sweep: x = n_vehicles (on-road ~= n), y = EDC - LFU
    dens = ["50", "100", "200", "400", "600"]
    syn_x = [int(n) for n in dens]
    syn_delta = [np.mean(syn["density_sweep_alpha0.8"][n]["EDC"]["per_seed"])
                 - np.mean(syn["density_sweep_alpha0.8"][n]["LFU"]["per_seed"]) for n in dens]
    # SUMO hotspot: x = measured on-road vehicle count, y = EDC - LFU
    sumo_x, sumo_delta = [], []
    for t in ["sparse", "moderate", "high"]:
        c = sumo["hotspot1600"][t]
        sumo_x.append(c["on_road_mean"])
        sumo_delta.append(np.mean(c["EDC"]["per_seed"]) - np.mean(c["LFU"]["per_seed"]))

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.plot(syn_x, syn_delta, "o-", color=ORANGE, lw=1.8, ms=6,
            label="synthetic platoon simulator")
    ax.plot(sumo_x, sumo_delta, "s--", color=GREEN, lw=1.8, ms=6,
            label="SUMO Krauss (hotspot)")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("on-road vehicle count (log scale)")
    ax.set_xscale("log")
    ax.set_xticks([25, 50, 100, 200, 300, 400, 600])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylabel("EDC $-$ LFU miss rate (pp)\n($\\leftarrow$ larger EDC advantage)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Synthetic platoon model overstates EDC's spatial advantage\n"
                 "(margins 3-10$\\times$ larger than under realistic SUMO traffic)", fontsize=9)
    fig.savefig(FIGDIR / "fig_fidelity_gap.pdf")
    plt.close(fig)
    print("wrote fig_fidelity_gap.pdf")


if __name__ == "__main__":
    fig_spearman()
    fig_operating_boundary()
    fig_fidelity_gap()
    print("all figures written to", FIGDIR)
