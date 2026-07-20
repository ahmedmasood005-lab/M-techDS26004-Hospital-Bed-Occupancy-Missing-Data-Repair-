"""Feature definitions and leakage tests."""
from __future__ import annotations
import unittest
import pandas as pd
from src.feature_engineering import engineer_features


class FeatureTests(unittest.TestCase):
    def test_features_and_past_only_lag(self) -> None:
        frame = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3), "hospital_name": ["A"]*3, "department": ["ICU"]*3, "season": ["Winter"]*3, "total_beds": [10]*3, "occupied_beds": [5, 7, 9], "reserved_beds": [1]*3, "available_beds": [4,2,0], "ICU_beds": [3]*3, "ICU_occupied": [1,2,3], "daily_admissions": [2,3,4], "daily_discharge": [1,2,3], "staff_on_duty": [5]*3, "nurses_on_duty": [3]*3, "doctors_on_duty": [1]*3, "emergency_cases": [2]*3, "emergency_beds": [2]*3, "weekend_flag": [0,0,1], "disease_outbreak_flag": [0,0,1]})
        result = engineer_features(frame)
        self.assertEqual(result.loc[1, "occupancy_lag_1"], 50)
        self.assertEqual(result.loc[2, "critical_occupancy_flag"], 1)
        self.assertIn("seasonal_demand_index", result)


if __name__ == "__main__": unittest.main()
