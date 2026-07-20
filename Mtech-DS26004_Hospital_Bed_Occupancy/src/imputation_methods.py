"""Fifteen reusable missing-value imputation strategies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import warnings
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer


METHODS = ["mean", "median", "mode", "forward_fill", "backward_fill", "linear_interpolation",
           "time_interpolation", "polynomial_interpolation", "knn", "iterative",
           "department_mean", "hospital_median", "seasonal", "rolling_median", "hybrid"]


@dataclass
class ImputationResult:
    frame: pd.DataFrame
    strategy: dict[str, str]


class ImputationEngine:
    """Apply scalar, temporal, grouped, multivariate, and hybrid repairs."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    @staticmethod
    def _fallback(series: pd.Series) -> pd.Series:
        return series.fillna(series.median() if pd.api.types.is_numeric_dtype(series) else series.mode().iloc[0])

    def apply(self, frame: pd.DataFrame, column: str, method: str, **params: object) -> pd.Series:
        """Return one fully imputed column while preserving observed values."""
        if column not in frame:
            raise KeyError(column)
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        series = frame[column].copy()
        if not series.isna().any():
            return series
        if method in {"mean", "median", "mode"}:
            strategy = "most_frequent" if method == "mode" else method
            values = SimpleImputer(strategy=strategy).fit_transform(series.to_frame()).ravel()
            return pd.Series(values, index=series.index, name=column)
        ordered = frame.assign(_date=pd.to_datetime(frame.get("date"), errors="coerce")).sort_values(
            [c for c in ["hospital_name", "department", "_date"] if c in frame or c == "_date"])
        work = ordered[column].copy()
        group_keys = [c for c in ["hospital_name", "department"] if c in ordered]
        grouped = ordered.groupby(group_keys, observed=True, sort=False, group_keys=False) if group_keys else None
        if method == "forward_fill":
            filled = grouped[column].transform(lambda s: s.ffill().bfill()) if grouped is not None else work.ffill().bfill()
        elif method == "backward_fill":
            filled = grouped[column].transform(lambda s: s.bfill().ffill()) if grouped is not None else work.bfill().ffill()
        elif method == "linear_interpolation":
            filled = grouped[column].transform(lambda s: s.interpolate(method="linear", limit_direction="both")) if grouped is not None else work.interpolate(method="linear", limit_direction="both")
        elif method == "time_interpolation":
            def interpolate_time(part: pd.DataFrame) -> pd.Series:
                temp = pd.Series(part[column].to_numpy(), index=part._date)
                values = temp.interpolate(method="time", limit_direction="both").to_numpy() if temp.index.notna().all() and not temp.index.duplicated().any() else part[column].interpolate(method="linear", limit_direction="both").to_numpy()
                return pd.Series(values, index=part.index)
            if grouped is not None:
                filled = work.copy()
                for index_values in grouped.indices.values():
                    labels = ordered.index[index_values]
                    filled.loc[labels] = interpolate_time(ordered.loc[labels])
            else:
                filled = interpolate_time(ordered)
        elif method == "polynomial_interpolation":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    operation = lambda s: s.interpolate(method="polynomial", order=int(params.get("order", 2)), limit_direction="both")
                    filled = grouped[column].transform(operation) if grouped is not None else operation(work)
                except (ValueError, ImportError):
                    filled = grouped[column].transform(lambda s: s.interpolate(method="linear", limit_direction="both")) if grouped is not None else work.interpolate(method="linear", limit_direction="both")
        elif method == "rolling_median":
            window = int(params.get("window", 7))
            rolling = grouped[column].transform(lambda s: s.rolling(window, center=True, min_periods=1).median()) if grouped is not None else work.rolling(window, center=True, min_periods=1).median()
            filled = work.fillna(rolling)
        elif method in {"department_mean", "hospital_median", "seasonal"}:
            group_col, agg = {"department_mean": ("department", "mean"),
                              "hospital_median": ("hospital_name", "median"),
                              "seasonal": ("season", "median")}[method]
            if group_col in ordered:
                grouped = ordered.groupby(group_col, observed=True)[column].transform(agg)
                filled = work.fillna(grouped)
            else:
                filled = work
        elif method in {"knn", "iterative"}:
            numeric = frame.select_dtypes(include=np.number).copy()
            usable = [c for c in numeric if numeric[c].notna().any()]
            numeric = numeric[usable]
            estimator = KNNImputer(n_neighbors=int(params.get("n_neighbors", 5)), weights="distance") if method == "knn" else IterativeImputer(
                random_state=self.random_state, max_iter=int(params.get("max_iter", 10)), sample_posterior=False)
            transformed = estimator.fit_transform(numeric)
            return pd.Series(transformed[:, usable.index(column)], index=frame.index, name=column)
        elif method == "hybrid":
            chosen = self.recommend(frame, column)
            return self.apply(frame, column, chosen, **params)
        else:
            raise AssertionError("Unreachable method")
        filled = self._fallback(filled)
        return filled.reindex(frame.index)

    def recommend(self, frame: pd.DataFrame, column: str) -> str:
        """Select a data-aware default; benchmark results can override this heuristic."""
        series = frame[column]
        if not pd.api.types.is_numeric_dtype(series):
            return "mode"
        missing = series.isna().mean()
        skew = abs(series.skew()) if series.notna().sum() > 2 else 0
        if column in {"occupancy_rate", "temperature", "daily_admissions", "daily_discharge"} and "date" in frame:
            return "time_interpolation"
        if column in {"staff_on_duty", "nurses_on_duty", "doctors_on_duty"} and "department" in frame:
            return "department_mean"
        if skew > 1 or missing > .2:
            return "hospital_median" if "hospital_name" in frame else "median"
        correlations = frame.select_dtypes(include=np.number).corr()[column].abs().drop(column, errors="ignore")
        if len(correlations) and correlations.max() > .65 and missing < .3:
            return "iterative"
        return "median"

    def repair(self, frame: pd.DataFrame, strategies: dict[str, str] | None = None) -> ImputationResult:
        """Repair all missing columns using supplied or hybrid strategies."""
        result = frame.copy()
        applied: dict[str, str] = {}
        for column in result.columns[result.isna().any()]:
            method = (strategies or {}).get(column, self.recommend(result, column))
            result[column] = self.apply(result, column, method)
            applied[column] = method
        return ImputationResult(result, applied)
