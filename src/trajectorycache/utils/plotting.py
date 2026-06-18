"""
Plotting helpers for experiment results.

Requires matplotlib (optional dependency); gracefully fails if absent.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _check_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        logger.warning("matplotlib not installed — plotting disabled")
        return False


def plot_hit_rates(
    per_step_data: Dict[str, List[float]],
    output_path: Optional[Path] = None,
    title: str = "Per-Step Hit Rate Comparison",
) -> None:
    """
    Line plot of per-step hit rates for multiple policies.

    Parameters
    ----------
    per_step_data : dict
        {policy_name: [hit_rate_per_step]}
    output_path : Path, optional
        Save figure here; if None, calls plt.show().
    """
    if not _check_matplotlib():
        return

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for policy, rates in per_step_data.items():
        ax.plot(rates, label=policy, alpha=0.8)

    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Hit Rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        logger.info("Figure saved to %s", output_path)
    else:
        plt.show()
    plt.close(fig)


def plot_bar_comparison(
    hit_rates: Dict[str, float],
    output_path: Optional[Path] = None,
    title: str = "Cache Hit Rate by Policy",
) -> None:
    """Bar chart comparing final hit rates across policies."""
    if not _check_matplotlib():
        return

    import matplotlib.pyplot as plt

    policies = list(hit_rates.keys())
    rates = [hit_rates[p] * 100 for p in policies]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(policies, rates, color=["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"])
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_ylabel("Hit Rate (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        logger.info("Bar chart saved to %s", output_path)
    else:
        plt.show()
    plt.close(fig)
