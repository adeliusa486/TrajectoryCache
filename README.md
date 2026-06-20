# TrajectoryCache

**A smarter, spatial edge cache for highly mobile vehicle networks (V2X / MEC)**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 Project Overview

When it comes to fast-moving vehicular networks (V2X), standard caching policies like LRU or LFU just can't keep up. Cars zoom in and out of a roadside unit's (RSU) range in seconds, which means looking at past requests isn't a great way to predict what's needed next.

**TrajectoryCache (TC)** is a lightweight caching algorithm built specifically for these kinds of environments. Instead of only looking backward, TC calculates a **spatial urgency signal** by looking at the real-time movement (position, speed, and heading) of cars heading toward the RSU.

By combining typical popularity metrics with real-time trajectory forecasting, TC cuts down cache misses during heavy, bursty traffic—saving crucial backhaul latency.


---

## ✨ Key Capabilities

- **Fast & Lightweight:** No heavy reinforcement learning loops here. The math is fast and simple ($O(|\mathcal{C}| \cdot |\mathcal{V}_r|)$), making it perfect for low-power edge hardware.
- **Built-in Traffic Simulators:** Test against independent, randomized traffic (like `SimPy`) or car-following platoons (like `SUMO`).
- **Plug-and-Play Caching:** Easily swap out `TrajectoryCache` for standard baselines (`LRU`, `LFU`, `FIFO`, `Random`) using a unified `BaseCache` interface.
- **Ready for Production:** Comes with a clean Python API and a full `FastAPI` REST backend for easy deployment.
- **Fully Reproducible:** Everything is deterministic. Our results are backed by rigorous multi-seed tests, so what you see is exactly what you get.

---

## 🧮 Architecture & Methodology

TrajectoryCache figures out what to drop from the cache by balancing two main signals:

1. **Spatial Urgency:** How soon nearby cars will actually hit the RSU's best coverage zone.
2. **Historical Popularity:** A sliding-window count of recent requests for a file.

The composite eviction score for a file $f$ looks like this:
$$ \text{Score}(f) = W \cdot \text{Urgency}(f) + (1 - W) \cdot \text{Popularity}(f) $$

Here, $W \in [0, 1]$ is a tunable knob you can adjust. If you set $W=0$, the algorithm just acts like a normalized LFU policy.

---

## 🚀 Getting Started

TrajectoryCache requires **Python 3.10+**. 

### Standard Installation

If you just want to run the core library and its basic dependencies:

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

If you want to run simulations, generate plots, or contribute to the code, you'll need the extra developer tools:

```bash
# This brings in pytest, black, ruff, mypy, matplotlib, seaborn, etc.
pip install -e ".[all]"
```

For strict, exact reproducibility of the published paper results, grab the lockfile version:
```bash
pip install -r requirements-lock.txt
pip install -e .
```

---

## 💻 Usage

You can use TrajectoryCache right in your Python code, through the command line, or hosted as a REST API.

### 1. Python Library

```python
from trajectorycache import TrajectoryCache, SimulationRunner, SimulationConfig

# Set up the spatial cache with a 20% urgency weight
cache = TrajectoryCache(capacity=20, urgency_weight=0.2)

# Set up a 200-vehicle highway simulation
config = SimulationConfig(
    n_steps=1000,
    n_vehicles=200,
    cache_capacity=20,
    seed=42,
)

# Run it
runner = SimulationRunner(cache=cache, config=config)
result = runner.run()

print(f"Hit rate achieved: {result.hit_rate:.2%}")
```

### 2. Command Line Interface (CLI)

We've bundled some handy CLI entrypoints for running benchmarks straight from your terminal.

```bash
# Run a quick benchmark comparing all policies
tc-benchmark --output experiments/results/
```

### 3. REST API & Docker Deployment

Need to deploy TrajectoryCache as a microservice on an RSU? No problem.

```bash
# Spin up the FastAPI dev server
tc-api
# Alternatively: make api
```
You can check out the interactive Swagger docs at `http://localhost:8000/docs`.

If you're deploying to production with Docker:
```bash
make docker-build
make docker-up
```

---

## ⚙️ Configuration

Experiments and API behavior are driven by YAML configs. You can tweak `configs/simulation.yaml` or just override things using environment variables (like `TC_N_VEHICLES=400`).

```yaml
road_length:       10000.0   # Total highway length in meters
n_vehicles:        200       # Total simulated vehicles
n_steps:           1000      # Simulation duration
cache_capacity:    20        # Number of files the edge can store
zipf_alpha:        0.8       # Content popularity skew
urgency_weight:    0.2       # Spatial urgency blending factor (W)
seed:              42        # PRNG seed
```

---

## 📊 Experimental Evaluation & Results

We put the system through its paces across 10 independent random seeds. All the numbers below map straight to our research paper.

### Main Performance (Table 1)
Evaluated on a 10 km highway with 200 vehicles, $\alpha=0.8$, and $W=0.2$.

| Policy         | SimPy (Independent Traffic) | SUMO (Platooning Traffic) |
|----------------|-----------------------------|---------------------------|
| **TrajectoryCache** | 54.51% ± 1.44%              | **52.05% ± 1.35%**        |
| LFU            | 53.32% ± 1.73%              | 52.85% ± 1.58%            |
| LRU            | 69.73% ± 2.28%              | 66.16% ± 2.02%            |
| FIFO           | 73.02% ± 1.75%              | 68.65% ± 1.71%            |

> [!NOTE]
> TrajectoryCache beats out all the baselines under bursty, platooning conditions (Wilcoxon $p=0.042$, one-sided). When traffic is completely uniform and independent, it performs just like LFU, which proves the urgency signal is specifically doing its job for realistic, clustered vehicle arrivals.

### Advanced Parameter Sweeps
We also included some scripts so you can test the algorithm's limits yourself:
- **Zipf Skew Sweep:** Test out $\alpha=0.5$ (lower skew) using `scripts/run_multiseed.py --zipf-alpha 0.5`.
- **Weight Ablation ($W$):** See why urgency needs to blend with popularity rather than replace it using `scripts/run_wsweep.py`.
- **Density Boundary:** Find the saturation point (400+ vehicles) where the spatial signals start giving diminishing returns using `scripts/run_density_sweep.py`.

---

## 🔬 Reproducibility Guide

If you want to regenerate every single result, stat, and figure from the paper from scratch, just run:

```bash
make pipeline
```

**Step-by-step reproduction:**
```bash
# 1. Generate full raw multi-seed JSON metrics
make results-alpha08
make results-alpha05

# 2. Compute Wilcoxon p-values and summary stats
make stats

# 3. Generate high-resolution PDF plots for the paper
make figures
```

**Determinism Guarantee:** Our `SimulationRunner` tightly controls Python's `random` module and NumPy's `np.random` states. As long as you use `requirements-lock.txt`, the results are 100% guaranteed to be deterministic. We actually enforce this in our CI pipeline via `tests/test_determinism.py`.

---

## 🗂️ Repository Structure

```text
TrajectoryCache/
├── src/trajectorycache/
│   ├── api/             # FastAPI endpoints and schemas
│   ├── cache/           # Core heuristic (trajectory.py) & baselines (lru.py, lfu.py)
│   ├── content/         # Zipf catalog generation
│   ├── evaluation/      # Benchmark orchestration
│   └── simulation/      # Highway kinematics (platoon vs independent)
├── scripts/             # CLI runners (run_multiseed.py, run_density_sweep.py)
├── experiments/
│   ├── results/         # Committed, reproducible JSON outputs
│   └── figures/         # Auto-generated PDF plots for LaTeX
├── configs/             # YAML config files
├── tests/               # Unit, integration, and determinism tests
└── paper/               # Final Elsevier LaTeX source and compiled PDF
```

---

## 🛠️ Development & Testing

We like to keep the codebase clean and reliable, so make sure to run these checks before submitting any PRs:

```bash
make format       # Auto-formats code with Black
make lint         # Lints codebase with Ruff
make type-check   # Verifies static types with MyPy
make test         # Runs full PyTest suite with coverage reporting
make smoke        # Runs a fast sanity check of the simulation loop
```

---

## 📝 Citation

If you use TrajectoryCache or its simulation framework in your own research, we'd appreciate a citation:

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

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for the legal details.

We love community contributions! Be sure to check out [CONTRIBUTING.md](CONTRIBUTING.md) for some quick guidelines on branching, submitting pull requests, and getting through the CI checks.
