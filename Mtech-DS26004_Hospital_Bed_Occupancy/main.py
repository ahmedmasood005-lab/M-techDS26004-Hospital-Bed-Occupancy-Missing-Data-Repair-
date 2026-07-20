"""Command-line orchestration for the complete data science lifecycle."""
from __future__ import annotations
import argparse
import joblib
import pandas as pd
from src.data_cleaner import clean_data
from src.data_generator import generate_dataset
from src.data_loader import load_dataset
from src.data_validator import DataValidator
from src.eda import create_charts
from src.feature_engineering import engineer_features
from src.imputation_benchmark import benchmark
from src.imputation_methods import ImputationEngine
from src.model_training import train_models
from src.report_generator import generate_report
from src.statistical_tests import run_statistical_tests
from src.utils import ROOT, configure_logging, ensure_directories, save_json

LOGGER = configure_logging()
RAW = ROOT / "data/raw/hospital_bed_occupancy_raw.csv"
CLEAN = ROOT / "data/processed/hospital_bed_occupancy_clean.csv"
REPAIRED = ROOT / "data/processed/hospital_bed_occupancy_repaired.csv"
FEATURES = ROOT / "data/processed/hospital_bed_occupancy_features.csv"


def prepare() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load/generate, clean, and hybrid-repair the dataset."""
    if not RAW.exists(): generate_dataset()
    clean = clean_data(load_dataset(RAW)); clean.to_csv(CLEAN, index=False)
    repair = ImputationEngine().repair(clean); repair.frame.to_csv(REPAIRED, index=False)
    save_json(repair.strategy, ROOT / "config/imputation_config.json")
    joblib.dump(repair.strategy, ROOT / "models/imputation_pipeline.joblib")
    return clean, repair.frame


def run_pipeline(fast: bool = True) -> None:
    """Generate every reproducible project artifact in dependency order."""
    ensure_directories(); generate_dataset(); clean, repaired = prepare()
    report = DataValidator().validate(clean); DataValidator().export(report)
    selected = ["occupancy_rate", "daily_discharge", "staff_on_duty", "nurses_on_duty", "infection_rate", "temperature", "ICU_occupied", "average_length_of_stay"]
    methods = ["mean", "median", "mode", "forward_fill", "linear_interpolation", "knn", "iterative", "department_mean", "hospital_median", "seasonal", "rolling_median", "hybrid"] if fast else None
    comparison = benchmark(clean, columns=selected if fast else None, methods=methods, seeds=(11, 29) if fast else (11, 29, 47))
    features = engineer_features(repaired).dropna(subset=["critical_occupancy_flag"]); features.to_csv(FEATURES, index=False)
    save_json(run_statistical_tests(repaired), ROOT / "outputs/reports/statistical_tests.json")
    train_models(features, fast=fast); create_charts(clean, repaired, comparison); generate_report(clean, comparison)
    LOGGER.info("Complete pipeline finished")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Hospital bed occupancy missing-data repair")
    actions = result.add_mutually_exclusive_group(required=True)
    for flag in ["generate-data", "validate-data", "run-eda", "benchmark-imputation", "engineer-features", "train-model", "generate-report", "run-all"]: actions.add_argument(f"--{flag}", action="store_true")
    result.add_argument("--full", action="store_true", help="Use all benchmark columns, methods, seeds, and model families")
    return result


def main() -> None:
    args = parser().parse_args(); ensure_directories()
    if args.generate_data: generate_dataset(); return
    clean, repaired = prepare()
    if args.validate_data: DataValidator().export(DataValidator().validate(clean))
    elif args.run_eda: create_charts(clean, repaired)
    elif args.benchmark_imputation: benchmark(clean, seeds=(11, 29, 47) if args.full else (11,))
    elif args.engineer_features: engineer_features(repaired).to_csv(FEATURES, index=False)
    elif args.train_model:
        features = load_dataset(FEATURES) if FEATURES.exists() else engineer_features(repaired); train_models(features, fast=not args.full)
    elif args.generate_report:
        path = ROOT / "outputs/experiments/imputation_comparison.csv"; generate_report(clean, pd.read_csv(path) if path.exists() else None)
    elif args.run_all: run_pipeline(fast=not args.full)


if __name__ == "__main__":
    main()
