"""Reusable rule-based validation for hospital census records."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from .utils import ROOT, save_json

VALID = {
    "hospital_type": {"Teaching", "Public", "Private"},
    "department": {"Emergency", "ICU", "Cardiology", "Medicine", "Surgery", "Pediatrics"},
    "season": {"Winter", "Spring", "Summer", "Autumn"},
}
NONNEGATIVE = ["total_beds", "occupied_beds", "available_beds", "reserved_beds", "ICU_beds",
               "ICU_occupied", "daily_admissions", "daily_discharge", "staff_on_duty",
               "nurses_on_duty", "doctors_on_duty"]


@dataclass
class ValidationIssue:
    rule: str
    severity: str
    count: int
    description: str


class DataValidator:
    """Run deterministic structural, domain, consistency, and outlier checks."""

    def validate(self, frame: pd.DataFrame) -> dict[str, Any]:
        issues: list[ValidationIssue] = []

        def add(rule: str, mask: pd.Series | np.ndarray, description: str, severity: str = "error") -> None:
            count = int(np.asarray(mask).sum())
            if count:
                issues.append(ValidationIssue(rule, severity, count, description))

        add("duplicate_rows", frame.duplicated(), "Exact duplicate records")
        add("missing_identifier", frame.get("record_id", pd.Series(index=frame.index, dtype=object)).isna(), "Missing record identifier")
        if "date" in frame:
            add("invalid_date", pd.to_datetime(frame.date, errors="coerce").isna(), "Invalid or missing date")
        for column in NONNEGATIVE:
            if column in frame:
                add(f"negative_{column}", frame[column].fillna(0) < 0, f"Negative values in {column}")
        if {"occupied_beds", "total_beds"} <= set(frame):
            add("occupied_exceeds_total", frame.occupied_beds > frame.total_beds, "Occupied beds exceed total beds")
        if {"ICU_occupied", "ICU_beds"} <= set(frame):
            add("icu_exceeds_capacity", frame.ICU_occupied > frame.ICU_beds, "ICU occupied exceeds ICU beds")
        if "available_beds" in frame:
            add("negative_available", frame.available_beds.fillna(0) < 0, "Available beds below zero")
        if "occupancy_rate" in frame:
            add("invalid_occupancy_rate", ~frame.occupancy_rate.between(0, 100) & frame.occupancy_rate.notna(), "Occupancy rate outside 0-100%")
        for column, labels in VALID.items():
            if column in frame:
                add(f"invalid_{column}", ~frame[column].isin(labels) & frame[column].notna(), f"Invalid labels in {column}")
        if {"nurses_on_duty", "doctors_on_duty", "staff_on_duty"} <= set(frame):
            add("impossible_staffing", (frame.nurses_on_duty + frame.doctors_on_duty) > frame.staff_on_duty * 1.5,
                "Nurse and doctor counts inconsistent with staff total", "warning")
        if {"total_beds", "occupied_beds", "reserved_beds", "available_beds"} <= set(frame):
            delta = (frame.total_beds - frame.occupied_beds - frame.reserved_beds - frame.available_beds).abs()
            add("bed_calculation", delta.fillna(0) > 1, "Bed balance is inconsistent")
        numeric = frame.select_dtypes(include=np.number)
        outliers = 0
        for column in numeric:
            series = numeric[column].dropna()
            if len(series) > 10:
                q1, q3 = series.quantile([.25, .75])
                outliers += int(((series < q1 - 3 * (q3 - q1)) | (series > q3 + 3 * (q3 - q1))).sum())
        if outliers:
            issues.append(ValidationIssue("extreme_outliers", "warning", outliers, "Values beyond 3 IQRs"))
        missing = int(frame.isna().sum().sum())
        penalties = sum(x.count * (2 if x.severity == "error" else .5) for x in issues)
        score = max(0.0, 100 - 100 * (missing + penalties) / max(frame.size, 1))
        return {"rows": len(frame), "columns": len(frame.columns), "missing_cells": missing,
                "missing_percentage": round(100 * missing / max(frame.size, 1), 3),
                "duplicate_rows": int(frame.duplicated().sum()), "quality_score": round(score, 2),
                "issues": [asdict(issue) for issue in issues]}

    def export(self, report: dict[str, Any], path: Path | None = None) -> Path:
        return save_json(report, path or ROOT / "outputs/reports/data_quality_report.json")
