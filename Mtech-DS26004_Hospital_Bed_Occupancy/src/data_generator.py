"""Generate realistic longitudinal hospital census data with mixed missingness."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import ROOT, configure_logging, seed_everything

LOGGER = configure_logging()
HOSPITALS = {
    "Apex Medical Center": ("Teaching", "Karachi", 1.18),
    "City General Hospital": ("Public", "Lahore", 1.05),
    "North Star Clinic": ("Private", "Islamabad", 0.88),
    "Riverbend Hospital": ("Public", "Multan", 0.96),
    "Unity Health Institute": ("Teaching", "Peshawar", 1.12),
}
DEPARTMENTS = {
    "Emergency": ("ER", 1.25), "ICU": ("ICU", 1.18),
    "Cardiology": ("Cardiac", 1.05), "Medicine": ("Medical", 1.00),
    "Surgery": ("Surgical", 0.94), "Pediatrics": ("Pediatric", 0.86),
}


def _season(month: pd.Series) -> pd.Series:
    return month.map({12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring",
                      5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer",
                      9: "Autumn", 10: "Autumn", 11: "Autumn"})


def generate_dataset(rows: int = 5400, seed: int = 42, output_path: Path | None = None) -> pd.DataFrame:
    """Create census rows with operational constraints and 10-25% targeted missingness."""
    if rows < 5000:
        raise ValueError("rows must be at least 5,000 for the project dataset")
    rng = seed_everything(seed)
    dates = pd.date_range("2024-01-01", periods=180, freq="D")
    hospitals = list(HOSPITALS)
    departments = list(DEPARTMENTS)
    base = pd.MultiIndex.from_product([dates, hospitals, departments], names=["date", "hospital_name", "department"]).to_frame(index=False)
    frame = base.iloc[:rows].copy()
    n = len(frame)
    frame.insert(0, "record_id", [f"HBO-{i:06d}" for i in range(1, n + 1)])
    frame["hospital_type"] = frame.hospital_name.map(lambda x: HOSPITALS[x][0])
    frame["city"] = frame.hospital_name.map(lambda x: HOSPITALS[x][1])
    frame["ward"] = frame.department.map(lambda x: DEPARTMENTS[x][0])
    frame["weekend_flag"] = (frame.date.dt.dayofweek >= 5).astype(int)
    frame["holiday_flag"] = ((frame.date.dt.dayofyear % 47 == 0) | (frame.date.dt.dayofyear.isin([1, 82, 227, 359]))).astype(int)
    frame["season"] = _season(frame.date.dt.month)
    frame["disease_outbreak_flag"] = (((frame.date.between("2024-01-20", "2024-02-18")) |
                                        frame.date.between("2024-08-05", "2024-08-28")) &
                                       frame.department.isin(["Emergency", "Medicine", "Pediatrics"])).astype(int)
    hospital_factor = frame.hospital_name.map(lambda x: HOSPITALS[x][2]).to_numpy()
    demand_factor = frame.department.map(lambda x: DEPARTMENTS[x][1]).to_numpy()
    seasonal = 1 + 0.10 * np.sin(2 * np.pi * frame.date.dt.dayofyear.to_numpy() / 365)
    outbreak = frame.disease_outbreak_flag.to_numpy()
    total_beds = np.round((rng.normal(45, 8, n) * hospital_factor * np.where(frame.department.eq("ICU"), .45, 1))).clip(10, 110)
    frame["total_beds"] = total_beds.astype(int)
    base_rate = np.clip(.66 * demand_factor * seasonal + .12 * outbreak + rng.normal(0, .065, n), .35, .98)
    frame["occupied_beds"] = np.minimum(total_beds, np.round(total_beds * base_rate)).astype(int)
    frame["reserved_beds"] = np.minimum(np.round(total_beds * rng.uniform(.02, .08, n)), total_beds - frame.occupied_beds).astype(int)
    frame["available_beds"] = (frame.total_beds - frame.occupied_beds - frame.reserved_beds).clip(lower=0)
    frame["emergency_beds"] = np.round(total_beds * np.where(frame.department.eq("Emergency"), .30, .06)).clip(1).astype(int)
    frame["ICU_beds"] = np.round(total_beds * np.where(frame.department.eq("ICU"), .75, .12)).clip(2).astype(int)
    frame["ICU_occupied"] = np.minimum(frame.ICU_beds, np.round(frame.ICU_beds * np.clip(base_rate + rng.normal(0, .07, n), .2, 1))).astype(int)
    frame["ventilators_available"] = np.maximum(0, frame.ICU_beds - frame.ICU_occupied + rng.integers(0, 4, n))
    admissions = rng.poisson(np.clip(6 * demand_factor * hospital_factor * seasonal + 4 * outbreak, 1, None))
    frame["daily_admissions"] = admissions
    frame["daily_discharge"] = np.maximum(0, admissions + rng.integers(-4, 5, n) - outbreak)
    frame["emergency_cases"] = rng.poisson(np.clip(8 * demand_factor + 7 * outbreak + 2 * (frame.season.eq("Winter")), 1, None))
    staffing_factor = 1 - .13 * frame.weekend_flag - .10 * frame.holiday_flag
    frame["staff_on_duty"] = np.maximum(4, np.round((frame.occupied_beds / 3.2 + 7) * staffing_factor + rng.normal(0, 2, n))).astype(int)
    frame["nurses_on_duty"] = np.maximum(2, np.round(frame.staff_on_duty * .62 + rng.normal(0, 1, n))).astype(int)
    frame["doctors_on_duty"] = np.maximum(1, np.round(frame.staff_on_duty * .20 + rng.normal(0, .6, n))).astype(int)
    frame["average_length_of_stay"] = np.clip(rng.gamma(2.2, 1.7, n) + 1.2 * frame.department.isin(["ICU", "Surgery"]), .5, 24).round(2)
    frame["patient_turnover_rate"] = (frame.daily_discharge / frame.total_beds * 100).round(2)
    frame["occupancy_rate"] = (frame.occupied_beds / frame.total_beds * 100).round(2)
    frame["mortality_rate"] = np.clip(rng.beta(1.4, 50, n) * 100 + 1.3 * frame.department.eq("ICU"), 0, 12).round(2)
    frame["infection_rate"] = np.clip(rng.beta(1.5, 38, n) * 100 + .8 * outbreak, 0, 15).round(2)
    city_temp = frame.city.map({"Karachi": 28, "Lahore": 25, "Islamabad": 21, "Multan": 27, "Peshawar": 23}).to_numpy()
    frame["temperature"] = (city_temp + 9 * np.sin(2 * np.pi * (frame.date.dt.dayofyear.to_numpy() - 90) / 365) + rng.normal(0, 2.2, n)).round(1)

    # MCAR, MAR, MNAR, department-specific, sequential, and block missingness.
    masks: dict[str, np.ndarray] = {}
    masks["temperature"] = rng.random(n) < .11
    masks["infection_rate"] = rng.random(n) < (.08 + .13 * frame.weekend_flag.to_numpy())
    masks["staff_on_duty"] = rng.random(n) < (.07 + .12 * frame.holiday_flag.to_numpy() + .04 * frame.weekend_flag.to_numpy())
    masks["nurses_on_duty"] = rng.random(n) < (.08 + .13 * frame.weekend_flag.to_numpy())
    masks["average_length_of_stay"] = rng.random(n) < (.07 + .12 * (frame.average_length_of_stay > 6).to_numpy())
    masks["mortality_rate"] = rng.random(n) < (.06 + .15 * (frame.mortality_rate > 4).to_numpy())
    masks["ventilators_available"] = rng.random(n) < (.06 + .16 * frame.department.eq("ICU").to_numpy())
    masks["daily_discharge"] = rng.random(n) < (.08 + .10 * frame.department.eq("Surgery").to_numpy())
    masks["emergency_cases"] = rng.random(n) < (.09 + .10 * frame.department.eq("Emergency").to_numpy())
    masks["ICU_occupied"] = rng.random(n) < (.08 + .08 * frame.department.eq("ICU").to_numpy())
    masks["patient_turnover_rate"] = rng.random(n) < .13
    masks["occupancy_rate"] = rng.random(n) < .10
    masks["daily_admissions"] = rng.random(n) < .10
    masks["doctors_on_duty"] = rng.random(n) < (.09 + .06 * frame.holiday_flag.to_numpy())
    for start in rng.integers(50, max(51, n - 50), 10):
        masks["occupancy_rate"][start:start + rng.integers(8, 24)] = True
    block = rng.choice(n, size=int(.03 * n), replace=False)
    for column in ["daily_admissions", "doctors_on_duty", "temperature"]:
        masks[column][block] = True
    for column, mask in masks.items():
        frame.loc[mask, column] = np.nan
    frame["date"] = frame.date.dt.strftime("%Y-%m-%d")
    output_path = output_path or ROOT / "data/raw/hospital_bed_occupancy_raw.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    LOGGER.info("Generated %s rows at %s (%.1f%% cells missing)", n, output_path, frame.isna().sum().sum() / frame.size * 100)
    return frame


if __name__ == "__main__":
    generate_dataset()
