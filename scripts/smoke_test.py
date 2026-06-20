#!/usr/bin/env python
"""
scripts/smoke_test.py

Lightweight end-to-end smoke test.
Validates: imports, config loading, simulation, API startup.
Writes SMOKE_TEST_REPORT.md to the project root.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS: list[dict[str, Any]] = []


def record(name: str, fn):
    """Run fn(), record pass/fail, return (passed, result)."""
    start = time.perf_counter()
    try:
        result = fn()
        elapsed = time.perf_counter() - start
        RESULTS.append(
            {"test": name, "status": "PASS", "elapsed_s": round(elapsed, 3), "detail": ""}
        )
        print(f"  [PASS] {name}  ({elapsed*1000:.1f} ms)")
        return True, result
    except Exception as exc:
        elapsed = time.perf_counter() - start
        tb = traceback.format_exc()
        RESULTS.append(
            {"test": name, "status": "FAIL", "elapsed_s": round(elapsed, 3), "detail": tb}
        )
        print(f"  [FAIL] {name}\n     {exc}")
        return False, None


def smoke_imports():
    import trajectorycache  # noqa
    from trajectorycache import (
        TrajectoryCache,
        LRUCache,
        LFUCache,
        RandomCache,
        FIFOCache,
        ContentCatalog,
        SimulationRunner,
        SimulationConfig,
        run_benchmark,
    )

    return "ok"


def smoke_cache_basic():
    from trajectorycache import TrajectoryCache

    c = TrajectoryCache(capacity=5, urgency_weight=0.5)
    for i in range(8):
        c.request(i % 5, float(i * 100), float(i))
    assert c.total_requests == 8
    assert len(c) <= 5
    return c.summary()


def smoke_config_load():
    from trajectorycache.utils.config import load_config

    cfg = load_config(ROOT / "configs" / "simulation.yaml")
    assert cfg.n_steps > 0
    return cfg


def smoke_simulation_run():
    from trajectorycache import TrajectoryCache, SimulationRunner, SimulationConfig

    cfg = SimulationConfig(
        n_steps=50, warmup_steps=10, n_vehicles=10, n_items=30, cache_capacity=8, seed=0
    )
    cache = TrajectoryCache(capacity=8, urgency_weight=0.4)
    runner = SimulationRunner(cache=cache, config=cfg)
    result = runner.run()
    assert result.total_requests > 0
    assert 0.0 <= result.hit_rate <= 1.0
    return result.to_dict()


def smoke_benchmark():
    # Note: This is a functional test only, not a paper result replication.
    from trajectorycache import run_benchmark
    from trajectorycache.simulation.runner import SimulationConfig

    cfg = SimulationConfig(
        n_steps=30, warmup_steps=5, n_vehicles=5, n_items=20, cache_capacity=5, seed=1
    )
    results = run_benchmark(config=cfg, verbose=False)
    assert len(results) >= 5
    return {k: round(v.hit_rate * 100, 2) for k, v in results.items()}


def smoke_api_startup():
    from fastapi.testclient import TestClient
    from trajectorycache.api.app import app

    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    return r.json()


def smoke_api_request():
    from fastapi.testclient import TestClient
    from trajectorycache.api.app import app

    client = TestClient(app)
    client.post("/cache/reset")
    payload = {
        "item_id": 1,
        "item_location": 500.0,
        "current_time": 0.0,
        "vehicles": [],
        "catalog": {},
    }
    r1 = client.post("/cache/request", json=payload)
    assert r1.json()["hit"] is False
    r2 = client.post("/cache/request", json={**payload, "current_time": 1.0})
    assert r2.json()["hit"] is True
    return "hit/miss cycle OK"


# ---------------------------------------------------------------------------


def main() -> None:
    print("\n" + "=" * 55)
    print("  TrajectoryCache  -  Smoke Test Suite")
    print("=" * 55 + "\n")

    tests = [
        ("Core imports", smoke_imports),
        ("Basic cache operation", smoke_cache_basic),
        ("Config loading (YAML)", smoke_config_load),
        ("Simulation run", smoke_simulation_run),
        ("Full benchmark", smoke_benchmark),
        ("API startup", smoke_api_startup),
        ("API request cycle", smoke_api_request),
    ]

    for name, fn in tests:
        record(name, fn)

    # -- Summary ----------------------------------------------------------
    passed = sum(1 for r in RESULTS if r["status"] == "PASS")
    failed = sum(1 for r in RESULTS if r["status"] == "FAIL")

    print(f"\n{'='*55}")
    print(f"  PASSED: {passed}/{len(RESULTS)}   FAILED: {failed}/{len(RESULTS)}")
    print(f"{'='*55}\n")

    # -- Write SMOKE_TEST_REPORT.md ----------------------------------------
    report_path = ROOT / "SMOKE_TEST_REPORT.md"
    lines = [
        "# Smoke Test Report\n",
        f"**Total:** {len(RESULTS)}  **Passed:** {passed}  **Failed:** {failed}\n\n",
        "| Test | Status | Time (ms) |",
        "|------|--------|-----------|",
    ]
    for r in RESULTS:
        icon = "[PASS]" if r["status"] == "PASS" else "[FAIL]"
        lines.append(f"| {r['test']} | {icon} {r['status']} | {r['elapsed_s']*1000:.1f} |")

    if failed:
        lines.append("\n## Failures\n")
        for r in RESULTS:
            if r["status"] == "FAIL":
                lines.append(f"### {r['test']}\n```\n{r['detail']}\n```\n")

    lines.append("\n## Environment\n")
    lines.append(f"- Python: {sys.version.split()[0]}")
    lines.append(
        "- All core modules imported successfully"
        if passed == len(RESULTS)
        else "- Some modules failed - see failures above"
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to: {report_path}\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
