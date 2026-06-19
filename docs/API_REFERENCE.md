# API Reference

## Python Library

### `TrajectoryCache`

```python
class TrajectoryCache(capacity, urgency_weight=0.5, pop_window=300.0,
                      t_pred=3.0, alpha_d=0.5, r_rel=500.0)
```

**Parameters**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `capacity` | int | — | Max items in cache (`C_max`) |
| `urgency_weight` | float | 0.5 | `W ∈ [0,1]`: weight of spatial urgency |
| `pop_window` | float | 300.0 | Sliding popularity window (seconds) |
| `t_pred` | float | 30.0 | Vehicle position lookahead horizon (seconds) |
| `alpha_d` | float | 0.1 | Urgency decay rate (s⁻¹) |
| `r_rel` | float | 800.0 | Relevance radius (metres) |

**Methods**

```python
.request(item_id, item_location, current_time,
         vehicles=None, catalog=None) -> bool
```
Process a content request. Returns `True` on hit, `False` on miss.

```python
.evict() -> Optional[int]
```
Force-evict the lowest-scoring item. Returns evicted `item_id` or `None`.

```python
.get_scores(vehicles, t, new_item=None) -> Dict[int, float]
```
Return composite scores for all cached items.

```python
.popularity_counts() -> Dict[int, int]
```
Sliding-window request counts.

```python
.summary() -> dict
.clear()
.reset_stats()
.hit_rate  .miss_rate  .hits  .misses  .total_requests
```

---

### `SimulationRunner`

```python
runner = SimulationRunner(cache: BaseCache, config: SimulationConfig)
result = runner.run(verbose=False) -> SimulationResult
```

**`SimulationConfig` fields**

| Field | Default | Description |
|-------|---------|-------------|
| `road_length` | 10000.0 | Highway length (m) |
| `n_vehicles` | 50 | Number of vehicles |
| `dt` | 1.0 | Time step (s) |
| `n_steps` | 1000 | Measurement steps |
| `warmup_steps` | 100 | Discarded warm-up steps |
| `cache_capacity` | 20 | Cache size |
| `n_items` | 200 | Catalog size |
| `zipf_alpha` | 1.2 | Popularity skew |
| `requests_per_step` | 5 | Requests per step |
| `seed` | 42 | RNG seed |

---

### `run_benchmark`

```python
from trajectorycache import run_benchmark

results = run_benchmark(
    config=SimulationConfig(...),   # optional
    policies=[...],                 # optional — defaults to all 5
    output_dir=Path("results/"),    # optional
    verbose=False,
) -> Dict[str, EvalMetrics]
```

---

### `build_cache`

```python
from trajectorycache import build_cache

cache = build_cache("trajectory", capacity=20, urgency_weight=0.4)
cache = build_cache("lru",        capacity=20)
cache = build_cache("lfu",        capacity=20)
cache = build_cache("random",     capacity=20)
cache = build_cache("fifo",       capacity=20)
```

---

## REST API

Base URL: `http://localhost:8000`

### `GET /health`
```json
{"status": "ok", "version": "0.1.0"}
```

### `GET /cache/status`
```json
{
  "summary": {"policy": "TrajectoryCache", "capacity": 20, "size": 7,
               "hits": 142, "misses": 58, "hit_rate": 71.0, "miss_rate": 29.0},
  "contents": {"3": {"location": 4200.5, "timestamp": 98.0, "access_count": 3}}
}
```

### `POST /cache/configure`
```json
{"capacity": 20, "urgency_weight": 0.5, "pop_window": 300.0,
 "t_pred": 3.0, "alpha_d": 0.5, "r_rel": 500.0}
```

### `POST /cache/request`
```json
{
  "item_id": 42,
  "item_location": 3500.0,
  "current_time": 120.0,
  "vehicles": [{"x": 3200.0, "speed": 25.0, "direction": 1}],
  "catalog": {"42": 3500.0, "7": 1200.0}
}
```
Response: `{"hit": true, "item_id": 42, "stats": {...}}`

### `POST /simulation/run`
```json
{"n_steps": 1000, "n_vehicles": 50, "cache_capacity": 20, "n_items": 200, "seed": 42}
```

### `GET /simulation/results`
Returns the last benchmark results as a dict keyed by policy name.
