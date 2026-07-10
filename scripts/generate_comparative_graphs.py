#!/usr/bin/env python3
"""
Generate Publication-Quality Comparative Graphs for TrajectoryCache vs. Baselines.
Produces high-resolution figures (300 DPI PNG + PDF) for:
1. Extended Policy Family Comparison (Bar chart with standard deviation error bars)
2. Vehicular Density Sweep & Crossover Boundary (Line plot)
3. Robustness under GNSS & V2X Telemetry Uncertainty (Grouped bar chart)
4. Wall-Clock Execution & Training Overhead (Log-scale bar plot)
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 2,
    'lines.markersize': 7,
    'grid.linestyle': '--',
    'grid.alpha': 0.6
})

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "figures")
PAPER_FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PAPER_FIG_DIR, exist_ok=True)

def plot_policy_family_comparison():
    print("[1/4] Generating Figure 1: Extended Policy Family Comparison across 10 Seeds...")
    policies = [
        "ProximityCache\n(Pure Spatial)",
        "FIFO\n(Temporal)",
        "Random\n(Baseline)",
        "LRU\n(Temporal)",
        "QLearningCache\n(Learning TD)",
        "LFU\n(Temporal)",
        "TrajectoryCache\n(Hybrid Rule)"
    ]
    miss_rates = [77.92, 68.81, 68.46, 66.30, 53.42, 53.40, 52.54]
    std_devs = [1.23, 1.49, 1.81, 1.75, 1.48, 1.50, 1.43]
    colors = ['#d95f02', '#7570b3', '#7570b3', '#7570b3', '#e7298a', '#1b9e77', '#2b5c8f']

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(policies, miss_rates, xerr=std_devs, color=colors, alpha=0.88, capsize=4, edgecolor='black')

    ax.set_xlabel('Mean Cache Miss Rate (%) [Lower is Better]')
    ax.set_title('Extended Comparative Evaluation Across Policy Families (Zipf α=0.8, 10 Seeds)')
    ax.set_xlim(45, 85)
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Annotate bars
    for bar, val, std in zip(bars, miss_rates, std_devs):
        ax.text(val + std + 0.6, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}% ± {std:.2f}%", va='center', ha='left', fontweight='bold' if val < 53 else 'normal')

    plt.tight_layout()
    for d in [OUTPUT_DIR, PAPER_FIG_DIR]:
        plt.savefig(os.path.join(d, "comparative_policy_family.png"), dpi=300)
        plt.savefig(os.path.join(d, "comparative_policy_family.pdf"))
    plt.close()
    print("      -> Saved comparative_policy_family.png/.pdf")

def plot_density_crossover():
    print("[2/4] Generating Figure 2: Vehicular Density Sweep & Empirical Crossover Boundary...")
    densities = [50, 100, 200, 400, 600]
    tc_miss = [46.25, 51.61, 52.54, 53.14, 53.59]
    lfu_miss = [53.27, 53.84, 53.40, 52.92, 52.87]
    q_miss = [53.35, 53.88, 53.45, 53.05, 52.95]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(densities, tc_miss, marker='s', color='#2b5c8f', label='TrajectoryCache (Rule-Based Hybrid)', linewidth=2.5)
    ax.plot(densities, lfu_miss, marker='o', color='#1b9e77', label='LFU (Classical Temporal)', linewidth=2)
    ax.plot(densities, q_miss, marker='^', color='#e7298a', linestyle='--', label='QLearningCache (Learning-Based TD)', linewidth=2)

    ax.axvline(x=300, color='gray', linestyle=':', alpha=0.8, label='Empirical Crossover (n ≈ 300)')
    ax.set_xlabel('Number of Simultaneously Active Vehicles (n)')
    ax.set_ylabel('Mean Cache Miss Rate (%)')
    ax.set_title('Sensitivity to Vehicular Density & Empirical Crossover Boundary')
    ax.set_xticks(densities)
    ax.grid(True)
    ax.legend(loc='lower right')

    plt.tight_layout()
    for d in [OUTPUT_DIR, PAPER_FIG_DIR]:
        plt.savefig(os.path.join(d, "comparative_density_crossover.png"), dpi=300)
        plt.savefig(os.path.join(d, "comparative_density_crossover.pdf"))
    plt.close()
    print("      -> Saved comparative_density_crossover.png/.pdf")

def plot_gnss_noise_sensitivity():
    print("[3/4] Generating Figure 3: Robustness under GNSS/V2X Positioning Noise & Update Lag...")
    conditions = [
        "Clean\n(0m, 0s)",
        "GNSS Noise\n(σ=5m)",
        "GNSS Noise\n(σ=15m)",
        "Telemetry Lag\n(1 step/1s)",
        "Telemetry Lag\n(3 steps/3s)",
        "Combined\n(15m + 3s lag)"
    ]
    tc_miss = [52.54, 52.50, 52.45, 52.70, 53.02, 53.04]
    q_miss = [53.42, 53.45, 53.48, 53.58, 53.84, 53.89]
    lfu_miss = [53.40, 53.40, 53.40, 53.40, 53.40, 53.40]

    x = np.arange(len(conditions))
    width = 0.26

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width, tc_miss, width, label='TrajectoryCache (Rule-Based)', color='#2b5c8f', edgecolor='black')
    ax.bar(x, q_miss, width, label='QLearningCache (Learning TD)', color='#e7298a', edgecolor='black')
    ax.bar(x + width, lfu_miss, width, label='LFU Reference (Telemetry Invariant)', color='#1b9e77', edgecolor='black')

    ax.set_ylabel('Mean Cache Miss Rate (%)')
    ax.set_title('Robustness Comparison Under GNSS Spatial Noise & V2X Telemetry Update Lag')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylim(50, 56)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='upper left')

    plt.tight_layout()
    for d in [OUTPUT_DIR, PAPER_FIG_DIR]:
        plt.savefig(os.path.join(d, "comparative_gnss_noise.png"), dpi=300)
        plt.savefig(os.path.join(d, "comparative_gnss_noise.pdf"))
    plt.close()
    print("      -> Saved comparative_gnss_noise.png/.pdf")

def plot_computational_overhead():
    print("[4/4] Generating Figure 4: Wall-Clock Per-Decision Inference Latency...")
    policies = ["LRU / FIFO", "LFU", "TrajectoryCache\n(Rule-Based O(C))", "QLearningCache\n(TD Learning)"]
    times_us = [1.2, 2.1, 38.4, 42.8]
    colors = ['#7570b3', '#1b9e77', '#2b5c8f', '#e7298a']

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(policies, times_us, color=colors, edgecolor='black', width=0.55)
    ax.set_ylabel('Execution Time per Cache Decision (μs)')
    ax.set_title('Wall-Clock Inference Overhead per Cache Eviction Decision')
    ax.set_ylim(0, 52)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for bar, val in zip(bars, times_us):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1.2, f"{val} μs", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    for d in [OUTPUT_DIR, PAPER_FIG_DIR]:
        plt.savefig(os.path.join(d, "comparative_computational_overhead.png"), dpi=300)
        plt.savefig(os.path.join(d, "comparative_computational_overhead.pdf"))
    plt.close()
    print("      -> Saved comparative_computational_overhead.png/.pdf")

def main():
    print("==========================================================================")
    print("  TrajectoryCache Comparative Benchmark Graph Generator")
    print("==========================================================================")
    plot_policy_family_comparison()
    plot_density_crossover()
    plot_gnss_noise_sensitivity()
    plot_computational_overhead()
    print("==========================================================================")
    print("All publication-quality graphs successfully generated in:")
    print(f"  - {OUTPUT_DIR}")
    print(f"  - {PAPER_FIG_DIR}")
    print("==========================================================================")

if __name__ == "__main__":
    main()
