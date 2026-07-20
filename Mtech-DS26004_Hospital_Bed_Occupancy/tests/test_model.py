"""Persisted model prediction and required output tests."""
from __future__ import annotations
import unittest
import joblib
import pandas as pd
from src.data_loader import load_dataset
from src.utils import ROOT, load_json


class ModelTests(unittest.TestCase):
    def test_model_prediction(self) -> None:
        model = joblib.load(ROOT / "models/critical_occupancy_model.joblib"); features = load_json(ROOT / "models/feature_names.json"); data = load_dataset(ROOT / "data/processed/hospital_bed_occupancy_features.csv")
        probability = model.predict_proba(data[features].head(2))[:, 1]
        self.assertEqual(len(probability), 2); self.assertTrue(((probability >= 0) & (probability <= 1)).all())

    def test_required_outputs_exist(self) -> None:
        required = [ROOT / "data/raw/hospital_bed_occupancy_raw.csv", ROOT / "data/processed/hospital_bed_occupancy_clean.csv", ROOT / "data/processed/hospital_bed_occupancy_features.csv", ROOT / "data/validation/masked_validation_data.csv", ROOT / "models/critical_occupancy_model.joblib", ROOT / "outputs/reports/Mtech_DS26004_Project_Report.pdf"]
        for path in required: self.assertTrue(path.exists(), str(path)); self.assertGreater(path.stat().st_size, 100)


if __name__ == "__main__": unittest.main()
