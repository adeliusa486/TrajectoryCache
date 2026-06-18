"""
Evaluation metrics for cache replacement experiments.

Computes: Hit Rate, Miss Rate, Byte Hit Rate, Average Latency,
          Backhaul Traffic, and per-policy comparison tables.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..simulation.runner import SimulationResult

logger = logging.getLogger(__name__)


@dataclass
class EvalMetrics:
    """Full metric suite for one policy run."""

    policy: str
    hit_rate: float
    miss_rate: float
    total_requests: int
    hits: int
    misses: int
    # Derived / optional
    mean_step_hit_rate: float = 0.0
    std_step_hit_rate: float = 0.0
    p5_step_hit_rate: float = 0.0
    p95_step_hit_rate: float = 0.0
    duration_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics(result: SimulationResult) -> EvalMetrics:
    """Derive full EvalMetrics from a SimulationResult."""
    shr = np.array(result.per_step_hit_rate) if result.per_step_hit_rate else np.array([0.0])
    return EvalMetrics(
        policy=result.policy,
        hit_rate=result.hit_rate,
        miss_rate=result.miss_rate,
        total_requests=result.total_requests,
        hits=result.hits,
        misses=result.misses,
        mean_step_hit_rate=float(np.mean(shr)),
        std_step_hit_rate=float(np.std(shr)),
        p5_step_hit_rate=float(np.percentile(shr, 5)),
        p95_step_hit_rate=float(np.percentile(shr, 95)),
        duration_s=result.duration_s,
    )


def compare_policies(metrics: List[EvalMetrics]) -> Dict[str, dict]:
    """
    Build a comparison dict ranked by hit_rate (descending).

    Returns
    -------
    dict
        {rank_1_policy: metrics_dict, rank_2_policy: metrics_dict, ...}
    """
    ranked = sorted(metrics, key=lambda m: m.hit_rate, reverse=True)
    return {m.policy: m.to_dict() for m in ranked}


def print_comparison_table(metrics: List[EvalMetrics]) -> None:
    """Pretty-print a comparison table to stdout."""
    header = f"{'Policy':<20} {'Hit Rate':>10} {'Miss Rate':>10} {'Requests':>10} {'Duration(s)':>12}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    ranked = sorted(metrics, key=lambda m: m.hit_rate, reverse=True)
    for m in ranked:
        print(
            f"{m.policy:<20} {m.hit_rate*100:>9.2f}% {m.miss_rate*100:>9.2f}% "
            f"{m.total_requests:>10} {m.duration_s:>11.3f}s"
        )
    print(sep)


def save_results(metrics: List[EvalMetrics], output_path: Path) -> None:
    """Persist metrics list to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [m.to_dict() for m in metrics]
    with open(output_path, "w") as fh:
        json.dump(data, fh, indent=2)
    logger.info("Results saved to %s", output_path)


def load_results(path: Path) -> List[EvalMetrics]:
    """Load previously saved metrics from JSON."""
    with open(path) as fh:
        data = json.load(fh)
    return [EvalMetrics(**d) for d in data]
