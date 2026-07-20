"""Repeated artificial masking benchmark with error, bias, distribution, and runtime metrics."""
from __future__ import annotations
import time
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

from .imputation_methods import ImputationEngine, METHODS
from .utils import ROOT, configure_logging

LOGGER = configure_logging()


def _metrics(actual: np.ndarray, predicted: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    mse = mean_squared_error(actual, predicted)
    safe = np.abs(actual) > 1e-8
    actual_std = np.std(actual)
    return {"mae": mean_absolute_error(actual, predicted), "rmse": np.sqrt(mse), "mse": mse,
            "r2": r2_score(actual, predicted) if len(actual) > 1 else np.nan,
            "mape": np.mean(np.abs(error[safe] / actual[safe])) * 100 if safe.any() else np.nan,
            "median_absolute_error": median_absolute_error(actual, predicted), "bias": np.mean(error),
            "percentage_bias": 100 * np.sum(error) / np.sum(actual) if np.sum(actual) else np.nan,
            "variance_distortion": abs(np.var(predicted) - np.var(actual)) / (np.var(actual) + 1e-8),
            "std_distortion": abs(np.std(predicted) - actual_std) / (actual_std + 1e-8),
            "ks_statistic": ks_2samp(actual, predicted).statistic,
            "wasserstein_distance": wasserstein_distance(actual, predicted),
            "correlation_preservation": np.corrcoef(actual, predicted)[0, 1] if len(actual) > 2 and np.std(predicted) else 0.0,
            "distribution_similarity": max(0.0, 1 - ks_2samp(actual, predicted).statistic)}


def benchmark(frame: pd.DataFrame, columns: Iterable[str] | None = None, methods: Iterable[str] | None = None,
              seeds: Iterable[int] = (11, 29, 47), mask_fraction: float = .10,
              output_path: Path | None = None) -> pd.DataFrame:
    """Benchmark methods per numeric column using repeated masked known values."""
    numeric = frame.select_dtypes(include=np.number).columns.tolist()
    columns = [c for c in (columns or numeric) if c in numeric and frame[c].notna().sum() >= 30]
    methods = list(methods or METHODS)
    engine = ImputationEngine()
    rows: list[dict[str, float | str | int]] = []
    masked_export: pd.DataFrame | None = None
    for column in columns:
        known = frame.index[frame[column].notna()].to_numpy()
        observed = frame.loc[known, column].to_numpy(float)
        for method in methods:
            for seed in seeds:
                rng = np.random.default_rng(seed)
                selected = rng.choice(known, size=max(10, int(mask_fraction * len(known))), replace=False)
                masked = frame.copy()
                actual = masked.loc[selected, column].to_numpy(float)
                masked.loc[selected, column] = np.nan
                if masked_export is None:
                    masked_export = masked.copy()
                    masked_export["masked_target_column"] = column
                    masked_export["masked_ground_truth"] = np.nan
                    masked_export.loc[selected, "masked_ground_truth"] = actual
                started = time.perf_counter()
                try:
                    predicted = engine.apply(masked, column, method).loc[selected].to_numpy(float)
                    elapsed = time.perf_counter() - started
                    metric = _metrics(actual, predicted, observed)
                    rows.append({"column": column, "method": method, "seed": seed, **metric, "runtime_seconds": elapsed})
                except Exception as exc:  # benchmark records failures without hiding them
                    LOGGER.warning("%s/%s failed: %s", column, method, exc)
    raw = pd.DataFrame(rows)
    if raw.empty:
        raise RuntimeError("No benchmark results were produced")
    metric_cols = [c for c in raw.select_dtypes(include=np.number) if c != "seed"]
    summary = raw.groupby(["column", "method"], as_index=False)[metric_cols].agg(["mean", "std"])
    summary.columns = ["_".join(filter(None, col)).rstrip("_") for col in summary.columns]
    for column, part in summary.groupby("column"):
        idx = part.index
        def scale(name: str) -> pd.Series:
            values = part[name].replace([np.inf, -np.inf], np.nan).fillna(part[name].max())
            span = values.max() - values.min()
            return (values - values.min()) / (span if span else 1)
        penalty = (.34 * scale("rmse_mean") + .18 * scale("mae_mean") + .14 * scale("bias_mean").abs() +
                   .12 * scale("ks_statistic_mean") + .08 * scale("variance_distortion_mean") +
                   .06 * scale("runtime_seconds_mean") + .08 * (1 - part.correlation_preservation_mean.clip(-1, 1)))
        summary.loc[idx, "final_score"] = (100 * (1 - penalty)).clip(0, 100)
        summary.loc[idx, "rank"] = summary.loc[idx, "final_score"].rank(ascending=False, method="min")
    summary["recommended_method"] = np.where(summary["rank"].eq(1), "Yes", "No")
    output_path = output_path or ROOT / "outputs/experiments/imputation_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    raw.to_csv(output_path.with_name("imputation_repeated_runs.csv"), index=False)
    if masked_export is not None:
        masked_export.to_csv(ROOT / "data/validation/masked_validation_data.csv", index=False)
    return summary.sort_values(["column", "rank"])
