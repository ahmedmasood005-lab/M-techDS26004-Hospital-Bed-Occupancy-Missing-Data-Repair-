"""Validation and missingness tests."""
from __future__ import annotations
import unittest
import pandas as pd
from src.data_validator import DataValidator
from src.missingness_analysis import missingness_summary


class ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame({"record_id": ["A", "B"], "date": ["2024-01-01", "bad"], "total_beds": [10, 10], "occupied_beds": [8, 12], "available_beds": [2, -2], "reserved_beds": [0, 0], "ICU_beds": [2, 2], "ICU_occupied": [1, 3], "occupancy_rate": [80.0, None]})

    def test_invalid_rules_detected(self) -> None:
        rules = {item["rule"] for item in DataValidator().validate(self.frame)["issues"]}
        self.assertTrue({"invalid_date", "occupied_exceeds_total", "icu_exceeds_capacity", "negative_available"} <= rules)

    def test_missing_values_detected(self) -> None:
        summary = missingness_summary(self.frame)
        self.assertEqual(summary.loc["occupancy_rate", "missing_count"], 1)


if __name__ == "__main__": unittest.main()
