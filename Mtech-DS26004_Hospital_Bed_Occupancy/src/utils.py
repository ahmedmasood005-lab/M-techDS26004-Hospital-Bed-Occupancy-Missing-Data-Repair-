"""Shared paths, logging, JSON, and reproducibility utilities."""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def ensure_directories() -> None:
    """Create every runtime output directory."""
    for path in (
        ROOT / "data/raw", ROOT / "data/processed", ROOT / "data/validation",
        ROOT / "models", ROOT / "database", ROOT / "outputs/charts",
        ROOT / "outputs/reports", ROOT / "outputs/predictions",
        ROOT / "outputs/experiments", ROOT / "outputs/screenshots",
    ):
        path.mkdir(parents=True, exist_ok=True)


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure console and rotating project-file logging."""
    ensure_directories()
    logger = logging.getLogger("hospital_occupancy")
    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        file_handler = logging.FileHandler(ROOT / "outputs/hospital_occupancy.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(stream)
        logger.addHandler(file_handler)
    return logger


def seed_everything(seed: int = 42) -> np.random.Generator:
    """Seed Python and NumPy and return a modern generator."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def load_json(path: Path) -> dict[str, Any]:
    """Load a UTF-8 JSON object."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(payload: Any, path: Path) -> Path:
    """Persist JSON, converting NumPy scalar values safely."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(value: Any) -> Any:
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Cannot serialize {type(value).__name__}")

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=default)
    return path
