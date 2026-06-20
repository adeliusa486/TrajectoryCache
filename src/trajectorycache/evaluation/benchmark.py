"""
Benchmark: run all cache policies under identical conditions and compare.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from ..cache import build_cache
from ..simulation.runner import SimulationConfig, SimulationResult, SimulationRunner
from .metrics import (
    EvalMetrics,
    compare_policies,
    compute_metrics,
    print_comparison_table,
    save_results,
)

logger = logging.getLogger(__name__)

# Default policies to benchmark (parameters match configs/simulation.yaml)
DEFAULT_POLICIES = [
    ("trajectory", {"urgency_weight": 0.2}),
    ("lfu", {"pop_window": 300.0}),
    ("lru", {}),
    ("random", {}),
    ("fifo", {}),
]


def run_benchmark(
    config: Optional[SimulationConfig] = None,
    policies: Optional[List[tuple]] = None,
    output_dir: Optional[Path] = None,
    verbose: bool = False,
) -> Dict[str, EvalMetrics]:
    """
    Run all policies under identical simulation conditions.

    Parameters
    ----------
    config : SimulationConfig, optional
    policies : list of (name, kwargs), optional
    output_dir : Path, optional
        If given, results are saved to JSON here.
    verbose : bool

    Returns
    -------
    dict
        {policy_name: EvalMetrics}
    """
    cfg = config or SimulationConfig()
    policy_list = policies or DEFAULT_POLICIES
    all_metrics: List[EvalMetrics] = []

    for policy_name, policy_kwargs in policy_list:
        logger.info("Benchmarking policy: %s", policy_name)
        cache = build_cache(policy_name, cfg.cache_capacity, **policy_kwargs)
        runner = SimulationRunner(cache=cache, config=cfg)
        result: SimulationResult = runner.run(verbose=verbose)
        metrics = compute_metrics(result)
        all_metrics.append(metrics)
        logger.info(
            "%s -> hit_rate=%.2f%%  miss_rate=%.2f%%",
            policy_name,
            metrics.hit_rate * 100,
            metrics.miss_rate * 100,
        )

    print_comparison_table(all_metrics)

    if output_dir:
        save_results(all_metrics, output_dir / "benchmark_results.json")

    return {m.policy: m for m in all_metrics}
