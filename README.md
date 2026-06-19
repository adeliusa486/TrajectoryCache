# TrajectoryCache

**Spatial-urgency-aware edge cache replacement for vehicular networks (V2X / MEC)**

[![CI](https://github.com/your-org/trajectorycache/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/trajectorycache/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

TrajectoryCache (TC) is a **mobility-aware content caching policy** for roadside edge servers in Vehicle-to-Everything (V2X) networks. Instead of evicting items purely by recency or frequency, TC jointly considers:

- **Spatial urgency** — how soon a nearby vehicle will need a given content item based on its current position, speed, and heading.
- **Historical popularity** — how frequently the item has been requested within a configurable sliding time-window.

The composite eviction score is:

```
Score(f) = W · Urgency(f) + (1 − W) · Popularity(f)
```

where `W ∈ [0, 1]` is a tunable weight. At `W=0`, TC reduces to normalised-LFU; at `W=1`, it becomes purely urgency-driven.

---

## Architecture

```
src/trajectorycache/
├── cache/
│   ├── base.py          ← Abstract BaseCache + CacheItem
│   ├── trajectory.py    ← TrajectoryCache (main algorithm)
│   ├── lru.py           ← LRU baseline
│   └── baselines.py     ← LFU, Random, FIFO baselines
├── simulation/
│   ├── highway.py       ← 1-D highway vehicle model
│   └── runner.py        ← SimulationRunner orchestrator
├── content/
│   └── catalog.py       ← Geo-tagged content + Zipf requests
├── evaluation/
│   ├── metrics.py       ← EvalMetrics, hit-rate stats
│   └── benchmark.py     ← Multi-policy benchmark runner
├── api/
│   └── app.py           ← FastAPI REST interface
└── utils/
    ├── config.py        ← YAML config loader
    ├── logging.py       ← Logging setup
    └── plotting.py      ← matplotlib helpers
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/your-org/trajectorycache.git
cd trajectorycache
pip install -e ".[dev]"
```

### Python API

```python
from trajectorycache import TrajectoryCache, SimulationRunner, SimulationConfig

# Create cache
cache = TrajectoryCache(capacity=20, urgency_weight=0.2)

# Configure simulation
cfg = SimulationConfig(
    n_steps=1000,
    n_vehicles=600,
    cache_capacity=20,
    seed=42,
)

# Run
runner = SimulationRunner(cache=cache, config=cfg)
result = runner.run()
print(f"Hit rate: {result.hit_rate:.2%}")
```

### Benchmark all policies

```bash
make benchmark
# or
python scripts/run_benchmark.py
```

### Hyperparameter sweep

```bash
python scripts/sweep.py --config configs/sweep.yaml
```

---

## Configuration

Edit `configs/simulation.yaml` or override with environment variables (`TC_<FIELD>=value`):

```yaml
road_length:       10000.0   # Total highway length (metres)
active_zone_length: 1600.0   # Active zone bounding content items (metres)
n_vehicles:        600       # Simulated vehicles (paper: 600)
n_steps:           1000
cache_capacity:    20
zipf_alpha:        0.8
urgency_weight:    0.2       # Optimal W from sweep
seed:              42
```

---

## Paper Results (Authentic Ground Truth)

The proposed TrajectoryCache has been rigorously evaluated across two distinct environments on a 10 km highway topology where content requests are localized within a dense 1.6 km active request zone:
1. **Independent Traffic (SimPy):** Baseline collision-free free-flow traffic.
2. **Platooning Traffic (SUMO):** Microscopic car-following dynamics governed by the Krauss model via SUMO 1.27.

### α = 0.8 (High Skew) — Mean Cache Miss Rate across 10 seeds

| Policy         | SimPy (Independent) | SUMO (Platooning) |
|----------------|---------------------|-------------------|
| LRU            | 80.24%              | 80.24%            |
| LFU            | 68.47%              | 67.58%            |
| **TC (W=0.2)** | **63.83%**          | **63.79%**        |

Under dense platooning traffic with the 1.6km bounded geographic constraint, TrajectoryCache achieves a strong ~63.8% miss rate, consistently outperforming the LFU baseline (67.58% - 68.47%). The algorithm natively thrives on the massive eviction battles induced by tight platoons approaching clustered content items.

---

## Project Structure

```

---

## Reproducing SUMO Results

The paper reports results under two simulation environments. The **kinematic pipeline** (`src/` + `scripts/`) is fully self-contained and reproducible from this repository. The **SUMO pipeline** requires external trace files.

### Kinematic pipeline (fully reproducible)

```bash
# Single-seed benchmark
python scripts/run_benchmark.py

# Multi-seed paper results (10 seeds, alpha=0.8 and alpha=0.5)
python scripts/run_multiseed.py

# W-parameter ablation sweep
python scripts/run_wsweep.py

# Vehicle density sweep
python scripts/run_density_sweep.py
```

### SUMO pipeline (requires trace files)

The SUMO Floating Car Data (FCD) traces used in the original paper (`fcd_{seed}.xml`,
one per seed) were generated with **SUMO 1.27** using the Krauss car-following model
on a straight 10 km highway network. These files are not included in the repository
due to size constraints.

To regenerate the traces:

1. Install SUMO 1.27: <https://sumo.dlr.de/>
2. Generate network and route files:
   ```bash
   netgenerate --straight --output-file=highway.net.xml --length=10000
   # Edit highway.rou.xml to set Krauss car-following model and 600 vehicles
   ```
3. Run SUMO for each seed to produce FCD output:
   ```bash
   sumo -c highway_seed{N}.sumocfg --fcd-output fcd_{N}.xml
   ```
4. Evaluate with the legacy script (note: contains outdated TC parameters — see deprecation warning):
   ```bash
   python sumo_cache_sim.py   # requires fcd_{seed}.xml files in working directory
   ```

> **Note:** `simpy_simulation.py` and `sumo_cache_sim.py` at the project root are
> **deprecated legacy scripts** with incorrect TC parameters (`GRZ_RADIUS=150`,
> `T_PREDICT=3.0`) that differ from the canonical `src/` implementation. They are
> retained for historical reference. All new experiments should use `scripts/`.

---

## Contributing

See `CONTRIBUTING.md`. PRs welcome!

## License

[MIT](LICENSE)
