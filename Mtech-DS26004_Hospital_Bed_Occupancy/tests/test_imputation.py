"""Scalar, KNN, and interpolation repair tests."""
from __future__ import annotations
import unittest
import numpy as np
import pandas as pd
from src.imputation_methods import ImputationEngine


class ImputationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=5), "hospital_name": ["A"]*5, "department": ["ICU"]*5, "x": [1., 2., np.nan, 4., 5.], "y": [2., 4., 6., 8., 10.]})

    def test_mean_and_median(self) -> None:
        engine = ImputationEngine(); self.assertEqual(engine.apply(self.frame, "x", "mean").iloc[2], 3); self.assertEqual(engine.apply(self.frame, "x", "median").iloc[2], 3)

    def test_knn(self) -> None:
        result = ImputationEngine().apply(self.frame, "x", "knn"); self.assertFalse(result.isna().any()); self.assertGreater(result.iloc[2], 2)

    def test_linear_interpolation(self) -> None:
        self.assertAlmostEqual(ImputationEngine().apply(self.frame, "x", "linear_interpolation").iloc[2], 3)


if __name__ == "__main__": unittest.main()
