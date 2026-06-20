"""Evaluation: metrics, benchmarking, and result reporting."""

from .benchmark import run_benchmark
from .metrics import (
    EvalMetrics,
    compare_policies,
    compute_metrics,
    print_comparison_table,
    save_results,
)

__all__ = [
    "run_benchmark",
    "EvalMetrics",
    "compute_metrics",
    "compare_policies",
    "print_comparison_table",
    "save_results",
]
