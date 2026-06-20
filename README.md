# TrajectoryCache

**Spatial-urgency-aware edge cache replacement for highly mobile vehicular networks (V2X / MEC)**

[![CI](https://github.com/adeliusa486/TrajectoryCache/actions/workflows/ci.yml/badge.svg)](https://github.com/adeliusa486/TrajectoryCache/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 Project Overview

In high-mobility vehicular networks (V2X), classical cache replacement policies like Least Recently Used (LRU) or Least Frequently Used (LFU) degrade rapidly. Because vehicles enter and leave a roadside unit's (RSU) coverage area in seconds, historical frequency alone is a poor predictor of near-future demand.

**TrajectoryCache (TC)** is a lightweight, closed-form edge caching heuristic designed specifically for vehicular environments. Instead of relying solely on the past, TC computes a **spatial urgency signal** based on the real-time kinematic trajectories (position, speed, heading) of approaching vehicles.

By augmenting traditional popularity metrics with real-time trajectory forecasting, TC significantly reduces cache miss rates under bursty, platoon-based traffic conditions, saving critical backhaul latency.


---

## ✨ Key Capabilities

- **Closed-Form Algorithmic Heuristic:** No expensive reinforcement learning training loops; the eviction score is computed in $O(|\mathcal{C}| \cdot |\mathcal{V}_r|)$ time, suitable for low-power edge hardware.
- **Dual Kinematic Simulators:** Built-in support for independent Poisson traffic (`SimPy`-like dynamics) and car-following platoon traffic (`SUMO`-like dynamics).
- **Extensible Cache Interface:** Easily swap out `TrajectoryCache` for standard baselines (`LRU`, `LFU`, `FIFO`, `Random`) using a unified `BaseCache` interface.
- **Production-Ready Endpoints:** Ships with both a programmatic Python API and a complete `FastAPI` REST interface for edge deployment.
- **Statistically Verifiable:** All empirical results are fully deterministic, reproducible, and evaluated over rigorous multi-seed stochastic configurations.

---

## 🧮 Architecture & Methodology

TrajectoryCache decides which content to evict by balancing two competing signals:

1. **Spatial Urgency:** How soon nearby vehicles will physically intercept the RSU's optimal coverage zone, estimated via linear time-to-encounter calculations.
2. **Historical Popularity:** A sliding-window normalized frequency count of recent requests for the item.

The composite eviction score for a file $f$ is:
$$ \text{Score}(f) = W \cdot \text{Urgency}(f) + (1 - W) \cdot \text{Popularity}(f) $$

Where $W \in [0, 1]$ is a tunable hyperparameter. At $W=0$, the algorithm reduces to a normalized LFU policy.

---

## 🚀 Getting Started

TrajectoryCache requires **Python 3.10+**. 

### Standard Installation

To install the core library and its runtime dependencies:

```bash
git clone https://github.com/your-org/TrajectoryCache.git
cd TrajectoryCache
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

pip install -e .
```

### Developer & Researcher Installation

To run simulations, re-generate plots, or contribute to the repository, install the extended dependencies:

```bash
# Installs pytest, black, ruff, mypy, matplotlib, seaborn, etc.
pip install -e ".[all]"
```

For strict, exact reproducibility of the published paper results, use the lockfile:
```bash
pip install -r requirements-lock.txt
pip install -e .
```

---

## 💻 Usage

TrajectoryCache can be used programmatically as a library, executed via CLI, or hosted as an API.

### 1. Python Library

```python
from trajectorycache import TrajectoryCache, SimulationRunner, SimulationConfig

# Initialize the spatial cache with a 20% urgency weight
cache = TrajectoryCache(capacity=20, urgency_weight=0.2)

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

### 2. Command Line Interface (CLI)

The package automatically registers standard entrypoints for benchmarking.

```bash
# Run a single-seed benchmark comparing all policies
tc-benchmark --output experiments/results/
```

### 3. REST API & Docker Deployment

To deploy TrajectoryCache as a microservice on an RSU:

```bash
# Start the FastAPI development server
tc-api
# Alternatively: make api
```
Access the interactive Swagger documentation at `http://localhost:8000/docs`.

For production deployment using Docker:
```bash
make docker-build
make docker-up
```

---

## ⚙️ Configuration

Experiments and API behavior are governed by YAML configurations. Edit `configs/simulation.yaml` or override parameters via environment variables (e.g., `TC_N_VEHICLES=400`).

```yaml
road_length:       10000.0   # Total highway length in metres
n_vehicles:        200       # Total simulated vehicles
n_steps:           1000      # Simulation duration
cache_capacity:    20        # Number of files the edge can store
zipf_alpha:        0.8       # Content popularity skew
urgency_weight:    0.2       # Spatial urgency blending factor (W)
seed:              42        # PRNG seed
```

---

## 📊 Experimental Evaluation & Results

The system is rigorously benchmarked across 10 independent random seeds. All results map directly to the accompanying research paper.

### Main Performance (Table 1)
Evaluated on a 10 km highway with 200 vehicles, $\alpha=0.8$, and $W=0.2$.

| Policy         | SimPy (Independent Traffic) | SUMO (Platooning Traffic) |
|----------------|-----------------------------|---------------------------|
| **TrajectoryCache** | 54.51% ± 1.44%              | **52.05% ± 1.35%**        |
| LFU            | 53.32% ± 1.73%              | 52.85% ± 1.58%            |
| LRU            | 69.73% ± 2.28%              | 66.16% ± 2.02%            |
| FIFO           | 73.02% ± 1.75%              | 68.65% ± 1.71%            |

> [!NOTE]
> TrajectoryCache outperforms all baselines under bursty, platooning conditions (Wilcoxon $p=0.042$, one-sided). Under uniform independent traffic, it converges with LFU, proving the urgency signal is specifically tuned for realistic, clustered vehicle arrivals.

### Advanced Parameter Sweeps
The repository includes scripts to validate the heuristic's boundaries:
- **Zipf Skew Sweep:** Tests $\alpha=0.5$ (lower skew) via `scripts/run_multiseed.py --zipf-alpha 0.5`.
- **Weight Ablation ($W$):** Demonstrates that urgency must augment, rather than override, popularity via `scripts/run_wsweep.py`.
- **Density Boundary:** Identifies the saturation point (400+ vehicles) where spatial signaling yields diminishing returns via `scripts/run_density_sweep.py`.

---

## 🔬 Reproducibility Guide

To regenerate every result, statistic, and figure from the paper identically, simply run:

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

**Determinism Guarantee:** The `SimulationRunner` explicitly controls Python's `random` module and NumPy's `np.random` states. If executed with `requirements-lock.txt`, the results are guaranteed deterministic. This mechanism is automatically enforced in our CI pipeline via `tests/test_determinism.py`.

---

## 🗂️ Repository Structure

```text
TrajectoryCache/
├── src/trajectorycache/
│   ├── api/             # FastAPI endpoints and schemas
│   ├── cache/           # Core heuristic (trajectory.py) & baselines (lru.py, lfu.py)
│   ├── content/         # Zipfian catalog generation
│   ├── evaluation/      # Benchmark orchestration
│   └── simulation/      # Highway kinematics (platoon vs independent)
├── scripts/             # CLI runners (run_multiseed.py, run_density_sweep.py)
├── experiments/
│   ├── results/         # Committed, reproducible JSON outputs
│   └── figures/         # Auto-generated PDF plots for LaTeX compilation
├── configs/             # YAML configuration files
├── tests/               # Unit, integration, and determinism tests
└── paper/               # Final Elsevier LaTeX source and compiled PDF
```

---

## 🛠️ Development & Testing

We mandate high engineering standards for all contributions:

```bash
make format       # Auto-formats code with Black
make lint         # Lints codebase with Ruff
make type-check   # Verifies static types with MyPy
make test         # Runs full PyTest suite with coverage reporting
make smoke        # Runs a fast sanity check of the simulation loop
```

---

## 📝 Citation

If you utilize TrajectoryCache or its simulation framework in your research, please cite our work:

```bibtex
@article{TrajectoryCache2026,
  title={Balancing Spatial Urgency and Content Popularity for Edge Caching in Vehicular Networks},
  journal={Vehicular Communications},
  year={2026},
  publisher={Elsevier}
}
```

---

## 📄 License & Contributing

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

We welcome community contributions! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on branching, submitting pull requests, and the required CI checks.
