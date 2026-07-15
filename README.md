# Balancing Spatial Urgency and Content Popularity for Edge Caching in Vehicular Networks


[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Project Overview

This repository contains the official implementation of **SpatialUrgencyCache (SU)** — referred to as **SU** throughout the accompanying paper and formerly named *TrajectoryCache* — a lightweight cache replacement heuristic designed for vehicular edge networks. (The old class name `TrajectoryCache` remains importable as a backward-compatible alias.)

Traditional cache replacement policies such as Least Recently Used (LRU) or Least Frequently Used (LFU) assume a relatively stationary client population. In vehicular networks, vehicles enter and leave a roadside unit's coverage area in seconds, rendering historical frequency alone insufficient for predicting near-future demand.

SU addresses this limitation by computing a spatial urgency signal based on the real-time kinematic trajectories (position, speed, heading) of approaching vehicles. By augmenting traditional popularity metrics with real-time trajectory forecasting, SU changes cache miss rates under bursty, platoon-based traffic conditions; the accompanying measurement study characterizes exactly when this helps and when it does not.


---

## Key Capabilities

- **Closed-Form Algorithm:** The eviction score is computed in linear time relative to the cache size and relevant vehicles, avoiding the overhead of reinforcement learning training loops and making it suitable for low-power edge hardware.
- **Traffic Simulators:** Built-in support for independent Poisson traffic (SimPy-like dynamics) and car-following platoon traffic (SUMO-like dynamics).
- **Extensible Interface:** Drop-in replacements for standard baselines (LRU, LFU, FIFO, Random) using a unified BaseCache interface.
- **REST API:** Includes a programmatic Python API and a FastAPI REST interface for edge deployment.
- **Reproducibility:** All empirical results are fully deterministic and evaluated over rigorous multi-seed stochastic configurations.

---

## Architecture and Methodology

SU determines which content to evict by balancing two signals:

1. **Spatial Urgency:** How soon nearby vehicles will physically intercept the optimal coverage zone, estimated via linear time-to-encounter calculations.
2. **Historical Popularity:** A sliding-window normalized frequency count of recent requests for the item.

The composite eviction score for a file f is:

Score(f) = W * Urgency(f) + (1 - W) * Popularity(f)

Where W is a tunable hyperparameter between 0 and 1. At W = 0, the algorithm reduces to a normalized LFU policy.

---

## Installation

SU requires Python 3.10 or higher. 

### Standard Installation

To install the core library and its runtime dependencies:

```bash
git clone https://github.com/adeliusa486/TrajectoryCache.git
cd TrajectoryCache
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
# Windows: .venv\Scripts\activate

pip install -e .
```

### Developer Installation

To run simulations, regenerate plots, or compile the paper, install the extended dependencies:

```bash
pip install -e ".[all]"
```

For strict reproducibility of the published paper results, use the lockfile:
```bash
pip install -r requirements-lock.txt
pip install -e .
```

---

## Usage

SU can be used programmatically, executed via CLI, or hosted as an API.

### 1. Python Library

```python
from trajectorycache import SpatialUrgencyCache, SimulationRunner, SimulationConfig

# Initialize the spatial cache with a 20 percent urgency weight
cache = SpatialUrgencyCache(capacity=20, urgency_weight=0.2)

# Configure a 200-vehicle highway simulation scenario
config = SimulationConfig(
    n_steps=1000,
    n_vehicles=200,
    cache_capacity=20,
    seed=42,
)

# Execute the simulation
runner = SimulationRunner(cache=cache, config=config)
result = runner.run()

print(f"Hit rate achieved: {result.hit_rate:.2%}")
```

### 2. Command Line Interface

The package provides standard entrypoints for benchmarking.

```bash
tc-benchmark --output experiments/results/
```

### 3. REST API and Docker

To deploy SU as a microservice on a roadside unit:

```bash
tc-api
# Alternatively: make api
```

For production deployment using Docker:
```bash
make docker-build
make docker-up
```

---

## Configuration

Experiments and API behavior are governed by YAML configurations. Edit `configs/simulation.yaml` or override parameters via environment variables (e.g., `TC_N_VEHICLES=400`).

```yaml
road_length:       10000.0   # Total highway length in meters
n_vehicles:        200       # Total simulated vehicles
n_steps:           1000      # Simulation duration
cache_capacity:    20        # Number of files the edge can store
zipf_alpha:        0.8       # Content popularity skew
urgency_weight:    0.2       # Spatial urgency blending factor
seed:              42        # PRNG seed
```

---

## Experimental Results

The system is benchmarked across 10 independent random seeds. All results map directly to the accompanying research paper.

### Main Performance
Evaluated on a 10 km highway with 200 vehicles, popularity skew alpha=0.8, and urgency weight W=0.2.

| Policy         | SimPy (Independent Traffic) | SUMO (Platooning Traffic) |
|----------------|-----------------------------|---------------------------|
| **SU** | 54.51% +/- 1.44%              | **52.05% +/- 1.35%**        |
| LFU            | 53.32% +/- 1.73%              | 52.85% +/- 1.58%            |
| LRU            | 69.73% +/- 2.28%              | 66.16% +/- 2.02%            |
| FIFO           | 73.02% +/- 1.75%              | 68.65% +/- 1.71%            |

Under this original 10 km configuration (request radius 800 m), SU edges out LFU on platooning traffic (Wilcoxon p=0.042, one-sided) and converges with LFU under independent traffic. The accompanying paper is a controlled measurement study showing this margin is an artifact of the request-radius setting: under a realistic short radius the margin reverses and LFU wins. See the paper and `scripts/make_paper_figures.py` for the fair-protocol results.

---

## Reproducibility Guide

To regenerate every result, statistic, and figure from the paper identically, execute the provided makefile targets. 

### System Requirements (Reviewer Information)
- **Hardware:** A standard multi-core desktop CPU (e.g., Intel Core i5/i7 or AMD Ryzen) with at least 8 GB of RAM. No GPU or specialized AI accelerators are required.
- **Software:** Python 3.10+ running on Linux, macOS, or Windows (via WSL or native).
- **Execution Time:** The complete `make pipeline` sweeps through 10 stochastic seeds across all policies and density configurations. Execution takes approximately 10-15 minutes on a standard modern CPU.

### Full Execution Pipeline

```bash
make pipeline
```

**Step-by-step reproduction:**
```bash
# 1. Generate full raw multi-seed JSON metrics
make results-alpha08
make results-alpha05

# 2. Compute Wilcoxon p-values and summary statistics
make stats

# 3. Generate high-resolution PDF plots for the paper
make figures
```

The `SimulationRunner` explicitly controls Python's `random` module and NumPy's `np.random` states. Executing with `requirements-lock.txt` guarantees deterministic results. This mechanism is enforced in the CI pipeline via `tests/test_determinism.py`.

---

## Repository Structure

```text
TrajectoryCache/
├── src/trajectorycache/
│   ├── api/             # FastAPI endpoints and schemas
│   ├── cache/           # Core SU heuristic (trajectory.py: SpatialUrgencyCache) & baselines
│   ├── content/         # Zipf catalog generation
│   ├── evaluation/      # Benchmark orchestration
│   └── simulation/      # Highway kinematics (platoon vs independent)
├── scripts/             # CLI runners (e.g., compute_stats.py)
├── experiments/
│   ├── results/         # Committed, reproducible JSON outputs
│   └── figures/         # Auto-generated PDF plots
├── configs/             # YAML configuration files
├── tests/               # Unit, integration, and determinism tests
└── paper/               # Final Elsevier LaTeX source and compiled PDF
```


## License

This project is licensed under the MIT License. See the LICENSE file for details.
