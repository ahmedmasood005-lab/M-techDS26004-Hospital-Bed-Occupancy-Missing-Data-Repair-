"""Append-only experiment tracking."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import csv

from .utils import ROOT


def log_experiment(experiment: dict[str, Any], path: Path | None = None) -> Path:
    """Append a normalized experiment row to CSV."""
    path = path or ROOT / "outputs/experiments/experiment_log.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"experiment_id": experiment.get("experiment_id", f"EXP-{datetime.now(timezone.utc):%Y%m%d%H%M%S%f}"),
           "timestamp_utc": datetime.now(timezone.utc).isoformat(), "random_seed": experiment.get("random_seed", 42),
           "imputation_method": experiment.get("imputation_method", "hybrid"), "model_name": experiment.get("model_name", ""),
           "hyperparameters": str(experiment.get("hyperparameters", {})), "validation_score": experiment.get("validation_score", ""),
           "test_score": experiment.get("test_score", ""), "runtime_seconds": experiment.get("runtime_seconds", ""),
           "notes": experiment.get("notes", "")}
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    return path
