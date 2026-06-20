"""
TrajectoryCache REST API.

Endpoints
---------
GET  /health                  - liveness probe
GET  /cache/status            - current cache state + stats
POST /cache/request           - simulate a content request
POST /cache/reset             - clear cache + stats
POST /simulation/run          - run a full benchmark
GET  /simulation/results      - retrieve last benchmark results
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from trajectorycache import TrajectoryCache, build_cache
from trajectorycache.evaluation.benchmark import run_benchmark
from trajectorycache.simulation.runner import SimulationConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="TrajectoryCache API",
    description="Spatial-urgency-aware vehicular edge cache REST interface",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton cache (overrideable via /cache/configure)
_cache: TrajectoryCache = TrajectoryCache(capacity=20, urgency_weight=0.5)
_last_results: Optional[Dict[str, Any]] = None

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class VehicleSchema(BaseModel):
    x: float = Field(..., description="Position along highway (m)")
    speed: float = Field(..., ge=0, description="Speed (m/s)")
    direction: int = Field(1, ge=-1, le=1, description="+1 or -1")


class CacheRequestBody(BaseModel):
    item_id: int
    item_location: float
    current_time: float
    vehicles: List[VehicleSchema] = []
    catalog: Dict[int, float] = {}


class ConfigureBody(BaseModel):
    capacity: int = Field(20, ge=1, le=10_000)
    urgency_weight: float = Field(0.5, ge=0.0, le=1.0)
    pop_window: float = Field(300.0, gt=0)
    t_pred: float = Field(3.0, gt=0)
    alpha_d: float = Field(0.5, gt=0)
    r_rel: float = Field(500.0, gt=0)


class BenchmarkBody(BaseModel):
    n_steps: int = Field(500, ge=10, le=50_000)
    n_vehicles: int = Field(50, ge=1)
    cache_capacity: int = Field(20, ge=1)
    n_items: int = Field(200, ge=1)
    seed: Optional[int] = 42


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", tags=["System"])
def health() -> dict:
    return {"status": "ok", "version": "0.1.0"}


@app.get("/cache/status", tags=["Cache"])
def cache_status() -> dict:
    return {
        "summary": _cache.summary(),
        "contents": {
            str(iid): {
                "location": item.location,
                "timestamp": item.timestamp,
                "access_count": item.access_count,
            }
            for iid, item in _cache.items().items()
        },
    }


@app.post("/cache/configure", tags=["Cache"])
def configure_cache(body: ConfigureBody) -> dict:
    global _cache
    _cache = TrajectoryCache(
        capacity=body.capacity,
        urgency_weight=body.urgency_weight,
        pop_window=body.pop_window,
        t_pred=body.t_pred,
        alpha_d=body.alpha_d,
        r_rel=body.r_rel,
    )
    logger.info("Cache reconfigured: %s", body.dict())
    return {"status": "reconfigured", "config": body.dict()}


@app.post("/cache/request", tags=["Cache"])
def cache_request(body: CacheRequestBody) -> dict:
    vehicles = [v.dict() for v in body.vehicles]
    hit = _cache.request(
        item_id=body.item_id,
        item_location=body.item_location,
        current_time=body.current_time,
        vehicles=vehicles,
        catalog=body.catalog,
    )
    return {
        "hit": hit,
        "item_id": body.item_id,
        "stats": _cache.summary(),
    }


@app.post("/cache/reset", tags=["Cache"])
def reset_cache() -> dict:
    _cache.clear()
    return {"status": "reset"}


@app.post("/simulation/run", tags=["Simulation"])
def run_simulation(body: BenchmarkBody) -> dict:
    global _last_results
    cfg = SimulationConfig(
        n_steps=body.n_steps,
        n_vehicles=body.n_vehicles,
        cache_capacity=body.cache_capacity,
        n_items=body.n_items,
        seed=body.seed,
    )
    metrics = run_benchmark(config=cfg, verbose=False)
    _last_results = {name: m.to_dict() for name, m in metrics.items()}
    return {"status": "complete", "results": _last_results}


@app.get("/simulation/results", tags=["Simulation"])
def get_results() -> dict:
    if _last_results is None:
        raise HTTPException(
            status_code=404, detail="No results yet. POST /simulation/run first."
        )
    return {"results": _last_results}
