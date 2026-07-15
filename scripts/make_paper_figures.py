#!/usr/bin/env python3
"""Generate the three main-text figures for the vehicular-caching paper
directly from the stored per-seed result JSONs. No numbers are hard-coded:
everything is read from experiments/results/ so the figures are reproducible
and provably consistent with Tables 1-2 and the diagnosis section.

Figures produced (vector PDF, for LaTeX \\includegraphics):
  fig_miss_by_tier.pdf     - Table 1: miss rate of 8 policies x 3 mobility tiers
  fig_ablation_flip.pdf    - Table 2: SU-LFU margin as one config knob changes
  fig_signal_correlation.pdf - Diagnosis: Spearman rho of each signal vs demand
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "experiments", "results")
OUT = os.path.join(HERE, "..", "paper_figures")
os.makedirs(OUT, exist_ok=True)


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


# --- shared publication style (clean, colourblind-safe, no chartjunk) ---
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "axes.grid": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
C_SYNTH, C_SUMO, C_REAL = "#4C72B0", "#DD8452", "#55A868"   # tiers
C_WIN, C_LOSE = "#4C72B0", "#C44E52"                          # margin sign
C_POP, C_URG = "#55A868", "#C44E52"                           # signals

# display order: strongest policies first so SU sits next to LFU
POLICY_ORDER = ["EDC", "LFU", "SU", "LRU", "FIFO", "Random", "Proximity", "QLearning"]
JSON_NAME = {"SU": "TC_W0.2"}          # JSON stores SU under its old code name


def _get(pol_dict, disp):
    return pol_dict[JSON_NAME.get(disp, disp)]


# ============================ Figure 1 ============================
def fig_miss_by_tier():
    mt = load("matched_tiers_535m.json")
    real = load("real_ngsim_i80.json")["policies"]
    synth, sumo = mt["synthetic"], mt["sumo"]

    means = {t: [] for t in ("s", "u", "r")}
    stds = {t: [] for t in ("s", "u", "r")}
    for p in POLICY_ORDER:
        for tag, src in (("s", synth), ("u", sumo), ("r", real)):
            d = _get(src, p)
            means[tag].append(d["mean"])
            stds[tag].append(d["std"])

    x = np.arange(len(POLICY_ORDER))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    for off, tag, col, lab in ((-w, "s", C_SYNTH, "Synthetic"),
                               (0.0, "u", C_SUMO, "SUMO"),
                               (w, "r", C_REAL, "Real (NGSIM I-80)")):
        ax.bar(x + off, means[tag], w, yerr=stds[tag], capsize=2.5,
               color=col, edgecolor="#222222", linewidth=0.4,
               error_kw=dict(elinewidth=0.7, ecolor="#444444"), label=lab)
    # dashed reference at LFU synthetic mean
    lfu_ref = _get(synth, "LFU")["mean"]
    ax.axhline(lfu_ref, ls="--", lw=0.8, color="#888888", zorder=0)
    ax.text(len(x) - 0.4, lfu_ref + 0.6, "LFU baseline", fontsize=7,
            color="#666666", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(POLICY_ORDER, rotation=20, ha="right")
    ax.set_ylabel("Cache miss rate (%)")
    ax.set_ylim(45, 88)
    ax.set_title("Controlled configuration (535 m, $r_{\\mathrm{rel}}=150$ m), "
                 "10 seeds; lower is better", fontsize=8.5)
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=8,
              handlelength=1.2, columnspacing=1.2)
    fig.savefig(os.path.join(OUT, "fig_miss_by_tier.pdf"))
    plt.close(fig)
    print("fig_miss_by_tier.pdf  (LFU synth mean = %.2f)" % lfu_ref)


# ============================ Figure 2 ============================
def fig_ablation_flip():
    ca = load("config_ablation.json")
    rows = [("C0_orig_10km", "Original: 10 km, $r_{\\mathrm{rel}}{=}800$ m"),
            ("C1_unidirectional", "$+$ unidirectional flow"),
            ("C2_dispersed_content", "$+$ content dispersed"),
            ("C3_small_rrel", "$+$ radius $800{\\to}150$ m"),
            ("C4_short_road", "$+$ short road (535 m)"),
            ("C5_realistic_slow", "$+$ realistic slow speed")]
    labels, margins, errs = [], [], []
    for key, lab in rows:
        su = np.array(ca[key]["SU"]["per_seed"])     # ablation JSON uses "SU"
        lfu = np.array(ca[key]["LFU"]["per_seed"])
        diff = su - lfu                      # negative => SU beats LFU
        labels.append(lab)
        margins.append(diff.mean())
        errs.append(diff.std())

    y = np.arange(len(labels))[::-1]         # top row first
    colors = [C_LOSE if m > 0 else C_WIN for m in margins]
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    ax.barh(y, margins, xerr=errs, capsize=2.5, color=colors,
            edgecolor="#222222", linewidth=0.4,
            error_kw=dict(elinewidth=0.7, ecolor="#444444"))
    ax.axvline(0, color="#333333", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("SU $-$ LFU miss-rate margin (pp)")
    # annotate the sign flip
    flip_idx = [i for i, (k, _) in enumerate(rows) if k == "C3_small_rrel"][0]
    ax.annotate("sign flips here", xy=(margins[flip_idx], y[flip_idx]),
                xytext=(margins[flip_idx] + 1.1, y[flip_idx] + 0.15),
                fontsize=7.5, color="#C44E52",
                arrowprops=dict(arrowstyle="->", color="#C44E52", lw=0.8))
    ax.text(-0.98, len(labels) - 0.4, "SU wins", fontsize=7.5, color=C_WIN,
            ha="right", style="italic")
    ax.text(0.98, len(labels) - 0.4, "LFU wins", fontsize=7.5, color=C_LOSE,
            ha="left", style="italic")
    ax.set_title("One knob changed per row (synthetic tier, 10 seeds)",
                 fontsize=8.5)
    fig.savefig(os.path.join(OUT, "fig_ablation_flip.pdf"))
    plt.close(fig)
    print("fig_ablation_flip.pdf  margins = " +
          ", ".join("%.2f" % m for m in margins))


# ============================ Figure 3 ============================
def fig_signal_correlation():
    sc = load("ngsim_signal_correlation.json")
    order = [("popularity", "Popularity\\n(LFU signal)"),
             ("pop_x_exposure", "Pop.$\\times$exposure\\n(EDC signal)"),
             ("urgency_SU", "Urgency\\n(SU signal)"),
             ("exposure", "Exposure")]
    labels = [lab for _, lab in order]
    means = [sc[k]["mean"] for k, _ in order]
    stds = [sc[k]["std"] for k, _ in order]
    cols = [C_POP, C_POP, C_URG, C_URG]

    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    ax.bar(x, means, 0.62, yerr=stds, capsize=3, color=cols,
           edgecolor="#222222", linewidth=0.4,
           error_kw=dict(elinewidth=0.8, ecolor="#444444"))
    for xi, m, s in zip(x, means, stds):
        ax.text(xi, m + s + 0.025, "%.2f" % m, ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("\\n", "\n") for l in labels], fontsize=7.5)
    ax.set_ylabel("Spearman $\\rho$ vs realized 30 s demand")
    ax.set_ylim(0, 1.0)
    ax.set_title("Real NGSIM I-80 (5 seeds): urgency is informative\n"
                 "but weaker than popularity", fontsize=8.5)
    leg = [Patch(facecolor=C_POP, edgecolor="#222", label="popularity-based"),
           Patch(facecolor=C_URG, edgecolor="#222", label="spatial/urgency")]
    ax.legend(handles=leg, frameon=False, fontsize=7.5, loc="upper right")
    fig.savefig(os.path.join(OUT, "fig_signal_correlation.pdf"))
    plt.close(fig)
    print("fig_signal_correlation.pdf  rho = " +
          ", ".join("%.3f" % m for m in means))


if __name__ == "__main__":
    fig_miss_by_tier()
    fig_ablation_flip()
    fig_signal_correlation()
    print("done ->", os.path.normpath(OUT))
