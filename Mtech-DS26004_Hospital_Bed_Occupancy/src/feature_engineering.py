"""Leakage-aware operational and temporal feature engineering."""
from __future__ import annotations
import numpy as np
import pandas as pd

FEATURE_DOCUMENTATION = {
    "occupancy_rate": "occupied_beds / total_beds x 100",
    "ICU_occupancy_rate": "ICU_occupied / ICU_beds x 100",
    "bed_utilization_ratio": "(occupied + reserved) / total beds",
    "admission_discharge_ratio": "admissions / max(discharges, 1)",
    "staff_to_patient_ratio": "staff / max(occupied beds, 1)",
    "nurse_to_patient_ratio": "nurses / max(occupied beds, 1)",
    "doctor_to_patient_ratio": "doctors / max(occupied beds, 1)",
    "emergency_pressure_index": "emergency cases / max(emergency beds, 1)",
    "bed_shortage_flag": "available beds <= 5",
    "high_occupancy_flag": "occupancy >= 80%",
    "critical_occupancy_flag": "occupancy >= 90%",
    "rolling/lag features": "past-only values within hospital and department; shifted before rolling",
    "aggregate occupancy": "expanding past-only average by month, department, or hospital",
    "weekend_staffing_gap": "weekend x max(expected staff - actual, 0)",
    "outbreak_pressure_score": "outbreak x (admissions + emergency cases) / total beds",
    "seasonal_demand_index": "past-only seasonal mean occupancy / overall past mean",
}


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create operational ratios and strictly past-looking time features."""
    result = frame.copy()
    result["date"] = pd.to_datetime(result.date)
    safe_beds = result.total_beds.clip(lower=1)
    safe_occupied = result.occupied_beds.clip(lower=1)
    result["occupancy_rate"] = result.occupied_beds / safe_beds * 100
    result["ICU_occupancy_rate"] = result.ICU_occupied / result.ICU_beds.clip(lower=1) * 100
    result["bed_utilization_ratio"] = (result.occupied_beds + result.reserved_beds) / safe_beds
    result["admission_discharge_ratio"] = result.daily_admissions / result.daily_discharge.clip(lower=1)
    result["staff_to_patient_ratio"] = result.staff_on_duty / safe_occupied
    result["nurse_to_patient_ratio"] = result.nurses_on_duty / safe_occupied
    result["doctor_to_patient_ratio"] = result.doctors_on_duty / safe_occupied
    result["emergency_pressure_index"] = result.emergency_cases / result.emergency_beds.clip(lower=1)
    result["bed_shortage_flag"] = (result.available_beds <= 5).astype(int)
    result["high_occupancy_flag"] = (result.occupancy_rate >= 80).astype(int)
    result["critical_occupancy_flag"] = (result.occupancy_rate >= 90).astype(int)
    result = result.sort_values(["hospital_name", "department", "date"]).reset_index(drop=True)
    groups = result.groupby(["hospital_name", "department"], observed=True, sort=False)
    prior_occ = groups.occupancy_rate.shift(1)
    result["seven_day_rolling_occupancy"] = prior_occ.groupby([result.hospital_name, result.department]).transform(lambda s: s.rolling(7, min_periods=1).mean())
    result["fourteen_day_rolling_occupancy"] = prior_occ.groupby([result.hospital_name, result.department]).transform(lambda s: s.rolling(14, min_periods=1).mean())
    result["occupancy_lag_1"] = groups.occupancy_rate.shift(1)
    result["occupancy_lag_7"] = groups.occupancy_rate.shift(7)
    result["admissions_lag_1"] = groups.daily_admissions.shift(1)
    result["discharge_lag_1"] = groups.daily_discharge.shift(1)
    prior_count = result.groupby("hospital_name").cumcount()
    prior_sum = result.groupby("hospital_name").occupancy_rate.cumsum() - result.occupancy_rate
    result["hospital_average_occupancy"] = prior_sum / prior_count.replace(0, np.nan)
    result["department_average_occupancy"] = (result.groupby("department").occupancy_rate.cumsum() - result.occupancy_rate) / result.groupby("department").cumcount().replace(0, np.nan)
    month_key = result.date.dt.to_period("M").astype(str)
    result["monthly_average_occupancy"] = (result.groupby(month_key).occupancy_rate.cumsum() - result.occupancy_rate) / result.groupby(month_key).cumcount().replace(0, np.nan)
    expected = result.groupby("department", observed=True).staff_on_duty.transform("median")
    result["weekend_staffing_gap"] = result.weekend_flag * (expected - result.staff_on_duty).clip(lower=0)
    result["outbreak_pressure_score"] = result.disease_outbreak_flag * (result.daily_admissions + result.emergency_cases) / safe_beds
    seasonal_past = (result.groupby("season", observed=True).occupancy_rate.cumsum() - result.occupancy_rate) / result.groupby("season", observed=True).cumcount().replace(0, np.nan)
    overall_past = (result.occupancy_rate.cumsum() - result.occupancy_rate) / pd.Series(np.arange(len(result)), index=result.index).replace(0, np.nan)
    result["seasonal_demand_index"] = seasonal_past / overall_past
    return result.replace([np.inf, -np.inf], np.nan)
