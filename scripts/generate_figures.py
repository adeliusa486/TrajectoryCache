#!/usr/bin/env python3
"""
generate_figures.py — Generate all paper figures from canonical result JSON.

Usage:
    python scripts/generate_figures.py \
        --input experiments/results/alpha08/multiseed_alpha0.8.json \
        --wsweep-input experiments/results/wsweep/wsweep_results.json \
        --density-input experiments/results/density/density_sweep.json \
        --output experiments/figures/
"""

import argparse
import json
import pathlib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

POLICY_COLORS = {
    "TrajectoryCache": "#c46210",  # tcorange
    "LFU": "#278c3c",  # tcgreen
    "LRU": "#1f61ab",  # tcblue
    "FIFO": "#6428a0",  # tcpurple
    "Random": "#64646e",  # tcgray
}

SEEDS = list(range(1, 11))


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def fig_perseed(data: dict, out: pathlib.Path):
    tc = data["sumo"]["TrajectoryCache"]["per_seed"]
    lfu = data["sumo"]["LFU"]["per_seed"]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(
        SEEDS, lfu, "o-", color=POLICY_COLORS["LFU"], label="LFU", linewidth=1.1, markersize=4
    )
    ax.plot(
        SEEDS,
        tc,
        "s-",
        color=POLICY_COLORS["TrajectoryCache"],
        label="TrajectoryCache ($W\\!=\\!0.2$)",
        linewidth=1.1,
        markersize=4,
    )

    ax.set_xlabel("Seed Index", fontsize=9)
    ax.set_ylabel("Miss Rate (\\%)", fontsize=9)
    ax.set_xticks(SEEDS)
    ax.set_ylim(45, 60)
    ax.grid(True, linestyle=":", color="gray", alpha=0.4)
    ax.legend(fontsize=7, loc="upper right")
    ax.tick_params(labelsize=7)

    fig.tight_layout()
    outpath = out / "fig_perseed.pdf"
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Written: {outpath}")


def fig_miss_bar(data: dict, out: pathlib.Path):
    policies = ["LRU", "FIFO", "Random", "LFU", "TrajectoryCache"]
    simpy_means = [data["simpy"][p]["miss_rate_mean"] for p in policies]
    sumo_means = [data["sumo"][p]["miss_rate_mean"] for p in policies]
    simpy_stds = [data["simpy"][p]["miss_rate_std"] for p in policies]
    sumo_stds = [data["sumo"][p]["miss_rate_std"] for p in policies]

    x = np.arange(len(policies))
    w = 0.35
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.bar(
        x - w / 2,
        simpy_means,
        w,
        yerr=simpy_stds,
        label="Independent (SimPy)",
        color="#1f61ab",
        alpha=0.85,
        capsize=3,
    )
    ax.bar(
        x + w / 2,
        sumo_means,
        w,
        yerr=sumo_stds,
        label="Platooning (SUMO-like)",
        color="#c46210",
        alpha=0.85,
        capsize=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(["LRU", "FIFO", "Random", "LFU", "TC"], fontsize=8)
    ax.set_ylabel("Mean Cache Miss Rate (\\%)", fontsize=9)
    ax.set_ylim(40, 85)
    ax.legend(fontsize=7)
    ax.grid(axis="y", linestyle=":", color="gray", alpha=0.4)
    ax.tick_params(labelsize=7)

    fig.tight_layout()
    outpath = out / "fig_miss_bar.pdf"
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Written: {outpath}")


def fig_wsweep(data: dict, out: pathlib.Path):
    w_vals = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    means = [data["w_sweep"][str(w)]["mean"] for w in w_vals]
    stds = [data["w_sweep"][str(w)]["std"] for w in w_vals]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(
        w_vals,
        means,
        "s-",
        color=POLICY_COLORS["FIFO"],
        label="TrajectoryCache",
        linewidth=1.2,
        markersize=4,
    )

    # LFU baseline
    ax.plot(
        [0.05, 0.95],
        [data["lfu_mean"], data["lfu_mean"]],
        "--",
        color=POLICY_COLORS["LFU"],
        label=f"LFU Baseline ({data['lfu_mean']:.2f}\\%)",
        linewidth=1.5,
    )

    ax.set_xlabel("Urgency weight $W$", fontsize=9)
    ax.set_ylabel("Mean Cache Miss Rate (\\%)", fontsize=9)
    ax.set_xticks(w_vals)
    ax.set_ylim(50, 80)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, linestyle=":", color="gray", alpha=0.4)
    ax.tick_params(labelsize=7)

    fig.tight_layout()
    outpath = out / "fig_wsweep.pdf"
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Written: {outpath}")


def fig_density(data: dict, out: pathlib.Path):
    densities = data["densities"]
    tc_means = data["tc_means"]
    lfu_means = data["lfu_means"]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(
        densities,
        tc_means,
        "s-",
        color=POLICY_COLORS["FIFO"],
        label="TrajectoryCache ($W\\!=\\!0.2$)",
        linewidth=1.2,
        markersize=4,
    )
    ax.plot(
        densities,
        lfu_means,
        "o-",
        color=POLICY_COLORS["LFU"],
        label="LFU",
        linewidth=1.2,
        markersize=4,
    )

    ax.set_xlabel("Number of Vehicles $n$", fontsize=9)
    ax.set_ylabel("Mean Cache Miss Rate (\\%)", fontsize=9)
    ax.set_xticks(densities)
    ax.set_ylim(42, 58)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, linestyle=":", color="gray", alpha=0.4)
    ax.tick_params(labelsize=7)

    fig.tight_layout()
    outpath = out / "fig_density.pdf"
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Written: {outpath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to multiseed JSON")
    parser.add_argument(
        "--wsweep-input", default="experiments/results/wsweep/wsweep_results.json"
    )
    parser.add_argument(
        "--density-input", default="experiments/results/density/density_sweep.json"
    )
    parser.add_argument("--output", required=True, help="Output directory for figures")
    args = parser.parse_args()

    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    data_main = load(args.input)
    fig_perseed(data_main, out)
    fig_miss_bar(data_main, out)

    try:
        data_wsweep = load(args.wsweep_input)
        fig_wsweep(data_wsweep, out)
    except FileNotFoundError:
        print(f"Warning: {args.wsweep_input} not found. Skipping wsweep figure.")

    try:
        data_density = load(args.density_input)
        fig_density(data_density, out)
    except FileNotFoundError:
        print(f"Warning: {args.density_input} not found. Skipping density figure.")

    print("All figures generated.")


if __name__ == "__main__":
    main()
