"""Conservative cleaning that preserves missing values for benchmarking."""
from __future__ import annotations
import pandas as pd


def clean_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate, normalize categories, parse dates, and enforce hard bounds."""
    result = frame.drop_duplicates(subset=["record_id"]).copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    for column in ["hospital_name", "hospital_type", "city", "department", "ward", "season"]:
        if column in result:
            result[column] = result[column].astype("string").str.strip()
    numeric = result.select_dtypes(include="number").columns
    for column in numeric:
        result.loc[result[column] < 0, column] = pd.NA
    result["occupied_beds"] = result[["occupied_beds", "total_beds"]].min(axis=1)
    result["ICU_occupied"] = result[["ICU_occupied", "ICU_beds"]].min(axis=1)
    result["available_beds"] = (result.total_beds - result.occupied_beds - result.reserved_beds).clip(lower=0)
    return result.sort_values(["hospital_name", "department", "date"]).reset_index(drop=True)
