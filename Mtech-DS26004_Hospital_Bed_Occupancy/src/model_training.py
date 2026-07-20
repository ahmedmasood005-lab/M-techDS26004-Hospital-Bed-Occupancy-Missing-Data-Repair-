"""Leakage-safe model comparison and persistence for critical occupancy risk."""
from __future__ import annotations
import time
from pathlib import Path
from typing import Any
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .experiment_logger import log_experiment
from .model_evaluation import evaluate_classifier, save_evaluation_plots
from .utils import ROOT, save_json


def _xgboost() -> Any:
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=180, max_depth=4, learning_rate=.06, subsample=.85, colsample_bytree=.85,
                             eval_metric="logloss", random_state=42, n_jobs=1)
    except ImportError:
        return GradientBoostingClassifier(random_state=42)


def train_models(frame: pd.DataFrame, fast: bool = False) -> dict[str, Any]:
    """Compare seven classifiers and select by 60% recall + 40% ROC-AUC."""
    target = "critical_occupancy_flag"
    excluded = {target, "record_id", "date", "occupancy_rate", "occupied_beds", "available_beds",
                "high_occupancy_flag", "bed_shortage_flag", "bed_utilization_ratio",
                "staff_to_patient_ratio", "nurse_to_patient_ratio", "doctor_to_patient_ratio"}
    features = [c for c in frame.columns if c not in excluded]
    X, y = frame[features].copy(), frame[target].astype(int)
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, test_size=.25, stratify=y_trainval, random_state=42)
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in features if c not in numeric]
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ])
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1200, class_weight="balanced", random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=8, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=180, max_depth=12, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=140, learning_rate=.06, max_depth=3, random_state=42),
        "XGBoost": _xgboost(), "Support Vector Machine": CalibratedClassifierCV(SVC(C=1.2, class_weight="balanced", random_state=42), method="sigmoid", cv=3, ensemble=False),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=11, weights="distance"),
    }
    if fast:
        models = {k: models[k] for k in ["Logistic Regression", "Random Forest", "Gradient Boosting", "XGBoost"]}
    results: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}
    for name, model in models.items():
        started = time.perf_counter()
        pipe = Pipeline([("preprocessing", preprocessing), ("classifier", model)])
        pipe.fit(X_train, y_train)
        probability = pipe.predict_proba(X_valid)[:, 1]
        prediction = (probability >= .42).astype(int)
        metrics = evaluate_classifier(y_valid.to_numpy(), prediction, probability)
        business_score = .60 * metrics["recall_sensitivity"] + .40 * metrics["roc_auc"]
        elapsed = time.perf_counter() - started
        results.append({"model": name, "validation_business_score": business_score, "runtime_seconds": elapsed,
                        **{k: v for k, v in metrics.items() if isinstance(v, (int, float, np.number))}})
        fitted[name] = pipe
        log_experiment({"model_name": name, "validation_score": business_score, "runtime_seconds": elapsed,
                        "notes": "Validation threshold=0.42; score=0.6 recall + 0.4 ROC-AUC"})
    comparison = pd.DataFrame(results).sort_values("validation_business_score", ascending=False)
    best_name = comparison.iloc[0].model
    best = fitted[best_name]
    best.fit(X_trainval, y_trainval)
    test_prob = best.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= .42).astype(int)
    metrics = evaluate_classifier(y_test.to_numpy(), test_pred, test_prob)
    ROOT.joinpath("models").mkdir(exist_ok=True)
    joblib.dump(best, ROOT / "models/critical_occupancy_model.joblib")
    joblib.dump(best.named_steps["preprocessing"], ROOT / "models/preprocessing_pipeline.joblib")
    save_json(features, ROOT / "models/feature_names.json")
    transformed_names = best.named_steps["preprocessing"].get_feature_names_out().tolist()
    save_json(transformed_names, ROOT / "models/transformed_feature_names.json")
    try:
        import shap
        transformed = best.named_steps["preprocessing"].transform(X_trainval)
        background = transformed[: min(200, len(transformed))]
        sample = transformed[: min(500, len(transformed))]
        explainer = shap.Explainer(best.named_steps["classifier"], background, feature_names=transformed_names)
        shap_values = explainer(sample)
        importance = np.abs(shap_values.values).mean(axis=0)
        importance_frame = pd.DataFrame({"feature": transformed_names, "mean_absolute_shap": importance}).sort_values("mean_absolute_shap", ascending=False)
        importance_frame.to_csv(ROOT / "outputs/experiments/shap_feature_importance.csv", index=False)
        top = importance_frame.head(18).sort_values("mean_absolute_shap")
        fig, ax = plt.subplots(figsize=(9, 7)); ax.barh(top.feature.str.replace("numeric__", "").str.replace("categorical__", ""), top.mean_absolute_shap, color="#6D5BD0")
        ax.set(title=f"SHAP Feature Importance - {best_name}", xlabel="Mean absolute SHAP value", ylabel="Transformed feature")
        fig.tight_layout(); fig.savefig(ROOT / "outputs/charts/shap_summary.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    except Exception as exc:
        save_json({"status": "unavailable", "reason": str(exc)}, ROOT / "outputs/reports/shap_status.json")
    comparison.to_csv(ROOT / "outputs/experiments/model_comparison.csv", index=False)
    payload = {"best_model": best_name, "selection_rule": "0.60 recall + 0.40 ROC-AUC at validation threshold 0.42",
               "threshold": .42, "test_metrics": metrics, "features": features, "model_comparison": comparison.to_dict("records")}
    save_json(payload, ROOT / "outputs/reports/model_evaluation.json")
    save_evaluation_plots(y_test.to_numpy(), test_pred, test_prob)
    return payload
