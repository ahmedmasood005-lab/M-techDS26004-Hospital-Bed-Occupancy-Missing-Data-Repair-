"""Portable CSV loading and schema normalization."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a hospital CSV and parse its date column."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path, low_memory=False)
    if "date" in frame:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame
