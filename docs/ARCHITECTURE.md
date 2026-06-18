# Architecture

## System Overview

TrajectoryCache models a **Mobile Edge Computing (MEC)** roadside unit (RSU) that caches
geo-tagged content for passing vehicles. The edge server has a finite cache (`C_max` items)
and must decide which content to evict when the cache is full.

```
  [Vehicle stream]
        │  (kinematics: x, speed, direction)
        ▼
  ┌─────────────────────────────────────────────────┐
  │              Edge Server / RSU                  │
  │                                                 │
  │  ┌─────────────┐    ┌────────────────────────┐ │
  │  │  Content    │    │   TrajectoryCache      │ │
  │  │  Requests   │───▶│                        │ │
  │  └─────────────┘    │  Score(f) =            │ │
  │                     │    W·Urgency(f)        │ │
  │  ┌─────────────┐    │  + (1-W)·Popularity(f)│ │
  │  │  Vehicle    │───▶│                        │ │
  │  │  Telemetry  │    └────────────────────────┘ │
  │  └─────────────┘              │                │
  └─────────────────────────────────────────────────┘
                                  │ miss
                                  ▼
                         [Backhaul / Origin]
```

## Module Map

```
cache/
  base.py         Abstract BaseCache + CacheItem dataclass
  trajectory.py   TrajectoryCache: scoring, eviction, popularity window
  lru.py          LRU (OrderedDict-based O(1) access)
  baselines.py    LFU (frequency dict), Random, FIFO

simulation/
  highway.py      1-D continuous highway; vehicles wrap at boundaries
  runner.py       SimulationRunner: ties highway + catalog + cache together

content/
  catalog.py      ContentCatalog: geo-tagged items, Zipf request sampling

evaluation/
  metrics.py      EvalMetrics dataclass, compute_metrics(), compare_policies()
  benchmark.py    run_benchmark(): runs all policies, prints table, saves JSON

api/
  app.py          FastAPI app: /health, /cache/*, /simulation/*

utils/
  config.py       YAML loader with TC_ env-var overrides
  logging.py      Centralised logging setup
  plotting.py     matplotlib bar + line charts (optional dependency)
```

## Scoring Algorithm

### Raw Urgency  `U_raw(f)`

```
For each vehicle v:
  x̂_v = x_v + s_v · d_v · T_pred         # predicted position after T_pred seconds
  if |x̂_v − ℓ_f| ≤ r_rel:                # vehicle is approaching item f
      TTE  = |ℓ_f − x_v| / s_v            # time-to-encounter (seconds)
      u    = 1 / (1 + α_d · TTE)          # urgency contribution
      U_raw(f) += u
```

Parameters: `T_pred` (lookahead horizon), `r_rel` (relevance radius), `α_d` (decay rate).

### Normalisation

Both urgency and popularity are min-max normalised across the candidate set
`C⁺ = cached_items ∪ {new_item}` to ensure fair weighting regardless of scale.

### Eviction

When a new item arrives and the cache is full:

1. Score every item in `C⁺`.
2. Find `victim = argmin Score`.
3. If `Score(victim) < Score(new_item)`: evict victim, insert new item.
4. Else: discard new item (keep existing cache).

## Data Flow (Simulation)

```
SimulationRunner.run()
  └─ for each step:
       highway.step()          → vehicle_states (list of dicts)
       catalog.generate_requests(k) → [ContentItem, ...]
       for each item:
           cache.request(item_id, item_loc, t, vehicles, catalog_map)
               → hit or miss
               → on miss: _fetch_from_backhaul() → score + evict
```

## Configuration Hierarchy

```
SimulationConfig defaults
  ↑ overridden by
configs/simulation.yaml
  ↑ overridden by
TC_<FIELD> environment variables
  ↑ overridden by
CLI flags (--n-steps, --capacity, …)
```

## Extension Points

| Extension | Where |
|-----------|-------|
| New cache policy | Subclass `BaseCache`, add to `REGISTRY` in `cache/__init__.py` |
| New vehicle model | Replace/extend `HighwaySimulation` |
| New content distribution | Subclass `ContentCatalog`, override `_build_zipf_weights` |
| New metrics | Add to `evaluation/metrics.py` |
| SUMO integration | Implement `SUMOHighway` adaptor, feed vehicle dicts to `cache.request` |
