# TrajectoryCache 🚗📡

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
cache = TrajectoryCache(capacity=20, urgency_weight=0.5)

# Configure simulation
cfg = SimulationConfig(
    n_steps=1000,
    n_vehicles=50,
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
python scripts/run_benchmark.py --n-steps 1000 --capacity 20
```

### Hyperparameter sweep

```bash
python scripts/sweep.py --config configs/sweep.yaml
```

### Start REST API

```bash
make api
# → http://localhost:8000/docs
```

### Docker

```bash
docker compose up -d
curl http://localhost:8000/health
```

---

## REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness probe |
| `GET` | `/cache/status` | Current cache state + stats |
| `POST` | `/cache/configure` | Reconfigure cache parameters |
| `POST` | `/cache/request` | Simulate a content request |
| `POST` | `/cache/reset` | Clear cache |
| `POST` | `/simulation/run` | Run full benchmark |
| `GET` | `/simulation/results` | Retrieve last benchmark results |

Full interactive docs at **`/docs`** (Swagger UI) and **`/redoc`**.

---

## Configuration

Edit `configs/simulation.yaml` or override with environment variables (`TC_<FIELD>=value`):

```yaml
road_length:       10000.0   # metres
n_vehicles:        50
n_steps:           1000
cache_capacity:    20
zipf_alpha:        1.2
urgency_weight:    0.5       # W in [0,1]
seed:              42
```

---

## Testing

```bash
make test            # full suite with coverage
make test-unit       # unit tests only
make test-integration
make smoke           # quick smoke validation
```

---

## TrajectoryCache Algorithm Details

### Spatial Urgency

For each cached item `f` at highway position `ℓ_f`:

```
x̂_v = x_v + s_v · d_v · T_pred        (predicted vehicle position)
TTE(v,f) = |ℓ_f − x_v| / s_v          (time-to-encounter)
u(v,f)   = 1 / (1 + α_d · TTE(v,f))   (per-vehicle urgency)
U_raw(f) = Σ u(v,f)  [for vehicles within r_rel of x̂_v]
```

### Popularity

Sliding-window request count normalised by the maximum count in the candidate set C⁺.

### Eviction Decision

When the cache is full and a new item arrives, TC scores all items in `C⁺ = cached_items ∪ {new_item}`. The item with the lowest composite score is evicted (or the new item is discarded if it scores lowest).

---

## Benchmarks

Example results (`n_steps=1000, capacity=20, n_vehicles=50, zipf_alpha=1.2`):

| Policy | Hit Rate |
|--------|----------|
| TrajectoryCache (W=0.5) | **~55–65%** |
| LFU | ~55–63% |
| LRU | ~50–57% |
| FIFO | ~46–52% |
| Random | ~44–50% |

*Results vary with traffic density, content distribution, and hyperparameters.*

---

## Project Structure

```
trajectorycache/
├── src/trajectorycache/   ← Main package
├── tests/
│   ├── unit/              ← Unit tests
│   └── integration/       ← End-to-end tests
├── configs/               ← YAML configs
├── scripts/               ← CLI scripts
├── experiments/results/   ← Benchmark outputs
├── docs/                  ← Extended documentation
├── notebooks/             ← Jupyter exploration
├── deployment/k8s/        ← Kubernetes manifests
└── .github/workflows/     ← CI/CD
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). PRs welcome!

## License

[MIT](LICENSE)
