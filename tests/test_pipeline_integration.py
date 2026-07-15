"""
Integration test: verify full pipeline produces valid JSON with correct schema.
Runs a 2-seed mini-benchmark (fast; not for paper results).
"""

import json
import pathlib
import subprocess

import pytest


@pytest.mark.slow
def test_multiseed_produces_valid_json(tmp_path):
    result = subprocess.run(
        [
            "python",
            "scripts/run_multiseed.py",
            "--seeds",
            "42",
            "43",
            "--zipf-alpha",
            "0.8",
            "--output",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"run_multiseed.py failed:\n{result.stderr}"

    output_json = tmp_path / "multiseed_alpha0.8.json"
    assert output_json.exists()

    with open(output_json) as f:
        data = json.load(f)

    for cond in ("sumo", "simpy"):
        assert cond in data
        for policy in ("SU", "LFU", "LRU", "FIFO", "Random"):
            assert policy in data[cond]
            assert "per_seed" in data[cond][policy]
            assert "miss_rate_mean" in data[cond][policy]
            assert "miss_rate_std" in data[cond][policy]
            assert len(data[cond][policy]["per_seed"]) == 2


def test_compute_stats_runs_on_committed_json():
    json_path = "experiments/results/alpha08/multiseed_alpha0.8.json"
    if not pathlib.Path(json_path).exists():
        pytest.skip(f"{json_path} does not exist yet")

    result = subprocess.run(
        ["python", "scripts/compute_stats.py", "--input", json_path],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"compute_stats.py failed:\n{result.stderr}"
    assert "p=" in result.stdout or "p-value" in result.stdout.lower()
