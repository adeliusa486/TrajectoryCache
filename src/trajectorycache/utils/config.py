"""
Configuration loading and validation.

Supports both YAML file loading and dict-based construction.
All config keys map to SimulationConfig fields.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from ..simulation.runner import SimulationConfig

logger = logging.getLogger(__name__)


def load_config(path: Optional[Path] = None) -> SimulationConfig:
    """
    Load SimulationConfig from a YAML file.

    Falls back to defaults if path is None or file is missing.
    Environment variables with prefix ``TC_`` override file values.
    """
    raw: Dict[str, Any] = {}

    if path and Path(path).exists():
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        logger.info("Loaded config from %s", path)
    else:
        logger.info("No config file found; using defaults")

    # Apply env overrides: TC_N_VEHICLES=30 -> raw["n_vehicles"] = 30
    for key, value in os.environ.items():
        if key.startswith("TC_"):
            field = key[3:].lower()
            try:
                # Attempt numeric coercion
                if "." in value:
                    raw[field] = float(value)
                else:
                    raw[field] = int(value)
            except ValueError:
                raw[field] = value

    # Filter to valid SimulationConfig fields
    valid_fields = SimulationConfig.__dataclass_fields__.keys()
    filtered = {k: v for k, v in raw.items() if k in valid_fields}

    return SimulationConfig(**filtered)


def save_config(cfg: SimulationConfig, path: Path) -> None:
    """Persist a SimulationConfig to YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    import dataclasses

    data = dataclasses.asdict(cfg)
    with open(path, "w") as fh:
        yaml.dump(data, fh, default_flow_style=False)
    logger.info("Config saved to %s", path)
