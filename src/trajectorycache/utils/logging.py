"""Centralised logging configuration for TrajectoryCache."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
) -> None:
    """
    Configure root logger.

    Parameters
    ----------
    level : str
        Logging level string (DEBUG, INFO, WARNING, ERROR).
    log_file : Path, optional
        If given, also write logs to this file.
    fmt : str
        Log message format string.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=numeric_level,
        format=fmt,
        handlers=handlers,
        force=True,
    )
    logging.getLogger("trajectorycache").setLevel(numeric_level)
