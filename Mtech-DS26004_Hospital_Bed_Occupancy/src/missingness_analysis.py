"""Missing-data summaries, patterns, correlations, and mechanism heuristics."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def missingness_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ordered missing counts and percentages."""
    result = pd.DataFrame({"missing_count": frame.isna().sum(), "missing_percentage": frame.isna().mean() * 100})
    return result[result.missing_count > 0].sort_values("missing_percentage", ascending=False)


def grouped_missingness(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    """Compute missing percentages by a categorical/time grouping."""
    missing = frame.drop(columns=[group], errors="ignore").isna().astype(float)
    return missing.groupby(frame[group], observed=True).mean().mul(100)


def missingness_patterns(frame: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Count the most common row-level missingness signatures."""
    columns = frame.columns[frame.isna().any()].tolist()
    signature = frame[columns].isna().apply(lambda row: " | ".join(row.index[row].tolist()) or "Complete", axis=1)
    return signature.value_counts().head(top_n).rename_axis("pattern").reset_index(name="rows")


def mechanism_evidence(frame: pd.DataFrame) -> dict[str, Any]:
    """Flag associations suggesting MAR/MNAR; labels are diagnostic, not proof."""
    evidence: dict[str, Any] = {"interpretation": "Mechanisms are synthetic ground truth plus association heuristics; MNAR is not identifiable from observed data alone.", "columns": {}}
    for column in frame.columns[frame.isna().any()]:
        indicator = frame[column].isna().astype(int)
        associations: list[tuple[str, float]] = []
        for category in ["department", "hospital_name", "weekend_flag", "holiday_flag", "season"]:
            if category not in frame or frame[category].nunique(dropna=True) < 2:
                continue
            table = pd.crosstab(indicator, frame[category])
            if table.shape[0] == 2:
                p_value = float(chi2_contingency(table)[1])
                if p_value < .05:
                    associations.append((category, p_value))
        label = "MAR-like" if associations else "MCAR-like"
        if column in {"mortality_rate", "average_length_of_stay"}:
            label = "MNAR-designed"
        evidence["columns"][column] = {"likely_mechanism": label, "associations": associations}
    return evidence
