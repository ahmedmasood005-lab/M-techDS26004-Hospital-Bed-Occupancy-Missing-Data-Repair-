"""Comprehensive binary-classification metrics and plots."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, classification_report, confusion_matrix, f1_score,
                             log_loss, matthews_corrcoef, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from .utils import ROOT


def evaluate_classifier(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    """Calculate decision and probability metrics with hospital-risk priorities."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0), "specificity": tn / max(tn + fp, 1),
            "f1": f1_score(y_true, y_pred, zero_division=0), "roc_auc": roc_auc_score(y_true, y_prob),
            "pr_auc": average_precision_score(y_true, y_prob), "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "matthews_correlation_coefficient": matthews_corrcoef(y_true, y_pred), "log_loss": log_loss(y_true, y_prob, labels=[0, 1]),
            "brier_score": brier_score_loss(y_true, y_prob), "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0)}


def save_evaluation_plots(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, output_dir: Path | None = None) -> list[Path]:
    """Export confusion, ROC, precision-recall, and calibration figures."""
    output_dir = output_dir or ROOT / "outputs/charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    palette = {"blue": "#176BCE", "cyan": "#15B7C9", "orange": "#F59E0B", "navy": "#102A43"}
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5)); image = ax.imshow(cm, cmap="Blues")
    for (i, j), value in np.ndenumerate(cm): ax.text(j, i, str(value), ha="center", va="center", fontsize=14)
    ax.set(title="Critical Occupancy Confusion Matrix", xlabel="Predicted class", ylabel="Actual class", xticks=[0, 1], yticks=[0, 1]); fig.colorbar(image, ax=ax)
    path = output_dir / "confusion_matrix.png"; fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig); paths.append(path)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5)); ax.plot(fpr, tpr, color=palette["blue"], lw=2, label=f"ROC-AUC {roc_auc_score(y_true, y_prob):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#64748B", label="No-skill"); ax.set(title="ROC Curve - Critical Occupancy", xlabel="False positive rate", ylabel="True positive rate"); ax.legend(); ax.grid(alpha=.2)
    path = output_dir / "roc_curve.png"; fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig); paths.append(path)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5)); ax.plot(recall, precision, color=palette["cyan"], lw=2, label=f"PR-AUC {average_precision_score(y_true, y_prob):.3f}")
    ax.set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision"); ax.legend(); ax.grid(alpha=.2)
    path = output_dir / "precision_recall_curve.png"; fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig); paths.append(path)
    fraction, predicted = calibration_curve(y_true, y_prob, n_bins=8, strategy="quantile")
    fig, ax = plt.subplots(figsize=(7, 5)); ax.plot(predicted, fraction, "o-", color=palette["orange"], label="Model"); ax.plot([0, 1], [0, 1], "--", color="#334155", label="Ideal")
    ax.set(title="Probability Calibration", xlabel="Mean predicted probability", ylabel="Observed critical rate"); ax.legend(); ax.grid(alpha=.2)
    path = output_dir / "calibration_curve.png"; fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig); paths.append(path)
    return paths
