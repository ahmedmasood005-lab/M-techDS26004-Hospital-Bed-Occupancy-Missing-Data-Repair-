"""Professional static EDA, imputation, and model figures."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px

from .utils import ROOT
from .imputation_methods import ImputationEngine

PALETTE = ["#176BCE", "#15B7C9", "#6D5BD0", "#F59E0B", "#0F9D76", "#E14D5A"]


def _finish(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white"); plt.close(fig); return path


def create_charts(frame: pd.DataFrame, repaired: pd.DataFrame | None = None,
                  benchmark: pd.DataFrame | None = None, output_dir: Path | None = None) -> list[Path]:
    """Export analytical charts with honest scales, units, and restrained colors."""
    output_dir = output_dir or ROOT / "outputs/charts"; output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=.9); data = frame.copy(); data["date"] = pd.to_datetime(data.date)
    paths: list[Path] = []
    trend = data.groupby("date", as_index=False).occupancy_rate.mean()
    fig, ax = plt.subplots(figsize=(10, 5)); ax.plot(trend.date, trend.occupancy_rate, color=PALETTE[0], lw=2)
    ax.axhline(90, color=PALETTE[3], ls="--", label="Critical threshold (90%)"); ax.set(title="Average Occupancy Trend", xlabel="Census date", ylabel="Occupancy rate (%)"); ax.legend(); paths.append(_finish(fig, output_dir / "occupancy_trend.png"))
    interactive = px.line(trend, x="date", y="occupancy_rate", title="Interactive Average Occupancy Trend", labels={"date": "Census date", "occupancy_rate": "Occupancy rate (%)"}, template="plotly_white")
    interactive.add_hline(y=90, line_dash="dash", line_color=PALETTE[3], annotation_text="Critical threshold")
    interactive_path = output_dir / "occupancy_trend_interactive.html"; interactive.write_html(interactive_path, include_plotlyjs=True); paths.append(interactive_path)
    grouped = data.groupby("department", as_index=False).occupancy_rate.mean().sort_values("occupancy_rate")
    fig, ax = plt.subplots(figsize=(8, 5)); ax.barh(grouped.department, grouped.occupancy_rate, color=PALETTE[0]); ax.set_xlim(0, 100); ax.set(title="Average Occupancy by Department", xlabel="Occupancy rate (%)", ylabel="Department"); paths.append(_finish(fig, output_dir / "department_occupancy.png"))
    hospital = data.groupby("hospital_name", as_index=False).occupancy_rate.mean().sort_values("occupancy_rate")
    fig, ax = plt.subplots(figsize=(9, 5)); ax.barh(hospital.hospital_name, hospital.occupancy_rate, color=PALETTE[2]); ax.set_xlim(0, 100); ax.set(title="Average Occupancy by Hospital", xlabel="Occupancy rate (%)", ylabel="Hospital"); paths.append(_finish(fig, output_dir / "hospital_occupancy.png"))
    sample = data.sample(min(1200, len(data)), random_state=42)
    fig, ax = plt.subplots(figsize=(7, 6)); ax.scatter(sample.total_beds, sample.occupied_beds, alpha=.28, color=PALETTE[1], s=18); ax.plot([0, sample.total_beds.max()], [0, sample.total_beds.max()], "--", color="#334155"); ax.set(title="Total Beds vs Occupied Beds", xlabel="Total beds", ylabel="Occupied beds"); paths.append(_finish(fig, output_dir / "beds_scatter.png"))
    fig, ax = plt.subplots(figsize=(7, 6)); ax.scatter(sample.daily_admissions, sample.daily_discharge, alpha=.3, color=PALETTE[0], s=18); ax.set(title="Admissions vs Discharges", xlabel="Daily admissions", ylabel="Daily discharges"); paths.append(_finish(fig, output_dir / "admissions_discharge.png"))
    fig, ax = plt.subplots(figsize=(8, 5)); sns.boxplot(data=data, x="department", y="ICU_occupied", color=PALETTE[1], ax=ax); ax.tick_params(axis="x", rotation=30); ax.set(title="ICU Occupancy by Department", xlabel="Department", ylabel="ICU occupied beds"); paths.append(_finish(fig, output_dir / "icu_analysis.png"))
    missing = data.isna().mean().mul(100).sort_values(ascending=False); missing = missing[missing > 0]
    fig, ax = plt.subplots(figsize=(9, 6)); ax.barh(missing.index[::-1], missing.values[::-1], color=PALETTE[3]); ax.set(title="Missing Values by Column", xlabel="Missing cells (%)", ylabel="Column"); paths.append(_finish(fig, output_dir / "missing_percentage.png"))
    heat = data.sample(min(600, len(data)), random_state=42).isna().astype(int).T
    fig, ax = plt.subplots(figsize=(12, 7)); sns.heatmap(heat, cmap=["#E8EEF5", PALETTE[3]], cbar=False, yticklabels=True, ax=ax); ax.set(title="Missingness Matrix (Sampled Rows)", xlabel="Sample row", ylabel="Column"); paths.append(_finish(fig, output_dir / "missingness_heatmap.png"))
    numeric = data.select_dtypes(include=np.number); corr = numeric.corr().loc[numeric.columns[:18], numeric.columns[:18]]
    fig, ax = plt.subplots(figsize=(12, 10)); sns.heatmap(corr, cmap="vlag", center=0, square=True, ax=ax, cbar_kws={"shrink": .7}); ax.set(title="Pearson Correlation Heatmap"); paths.append(_finish(fig, output_dir / "correlation_heatmap.png"))
    fig, ax = plt.subplots(figsize=(8, 5)); sns.boxplot(data=data, x="season", y="occupancy_rate", order=["Winter", "Spring", "Summer", "Autumn"], color=PALETTE[2], ax=ax); ax.set(title="Seasonal Occupancy Pattern", xlabel="Season", ylabel="Occupancy rate (%)"); paths.append(_finish(fig, output_dir / "seasonal_occupancy.png"))
    fig, ax = plt.subplots(figsize=(7, 5)); sns.boxplot(data=data, x="weekend_flag", y="occupancy_rate", color=PALETTE[1], ax=ax); ax.set(title="Weekend vs Weekday Occupancy", xlabel="Weekend flag (0=weekday, 1=weekend)", ylabel="Occupancy rate (%)"); paths.append(_finish(fig, output_dir / "weekend_analysis.png"))
    fig, ax = plt.subplots(figsize=(7, 5)); sns.boxplot(data=data, x="disease_outbreak_flag", y="occupancy_rate", color=PALETTE[3], ax=ax); ax.set(title="Outbreak Impact on Occupancy", xlabel="Disease outbreak flag", ylabel="Occupancy rate (%)"); paths.append(_finish(fig, output_dir / "outbreak_impact.png"))
    fig, ax = plt.subplots(figsize=(7, 6)); ax.scatter(sample.staff_on_duty, sample.occupancy_rate, alpha=.3, color=PALETTE[4], s=18); ax.set(title="Staffing vs Occupancy", xlabel="Staff on duty", ylabel="Occupancy rate (%)"); paths.append(_finish(fig, output_dir / "staffing_occupancy.png"))
    if repaired is not None:
        column = "occupancy_rate"; fig, ax = plt.subplots(figsize=(9, 5)); sns.kdeplot(data[column].dropna(), label="Before repair", color=PALETTE[3], ax=ax); sns.kdeplot(repaired[column].dropna(), label="After repair", color=PALETTE[0], ax=ax); ax.set(title="Occupancy Distribution Before and After Repair", xlabel="Occupancy rate (%)"); ax.legend(); paths.append(_finish(fig, output_dir / "distribution_before_after.png"))
        fig, ax = plt.subplots(figsize=(7, 5)); ax.boxplot([data[column].dropna(), repaired[column].dropna()], tick_labels=["Before", "After"], patch_artist=True); ax.set(title="Occupancy Boxplot Before and After Repair", ylabel="Occupancy rate (%)"); paths.append(_finish(fig, output_dir / "boxplot_before_after.png"))
        known = data.index[data[column].notna()].to_numpy(); rng = np.random.default_rng(73); selected = rng.choice(known, size=min(500, max(20, len(known)//10)), replace=False)
        masked = data.copy(); actual = masked.loc[selected, column].to_numpy(float); masked.loc[selected, column] = np.nan
        predicted = ImputationEngine().apply(masked, column, "hybrid").loc[selected].to_numpy(float); residual = predicted - actual
        fig, ax = plt.subplots(figsize=(7, 6)); ax.scatter(actual, predicted, alpha=.35, color=PALETTE[0], s=22); limits = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]; ax.plot(limits, limits, "--", color="#334155", label="Ideal"); ax.set(title="Actual vs Imputed Occupancy", xlabel="Actual occupancy rate (%)", ylabel="Imputed occupancy rate (%)"); ax.legend(); paths.append(_finish(fig, output_dir / "actual_vs_imputed.png"))
        fig, ax = plt.subplots(figsize=(7, 5)); ax.hist(residual, bins=28, color=PALETTE[2], edgecolor="white"); ax.axvline(0, color="#334155", ls="--"); ax.set(title="Imputation Residual Distribution", xlabel="Imputed - actual occupancy (percentage points)", ylabel="Masked observations"); paths.append(_finish(fig, output_dir / "residual_distribution.png"))
    if benchmark is not None and not benchmark.empty:
        best_column = benchmark.column.value_counts().index[0]; subset = benchmark[benchmark.column.eq(best_column)].sort_values("rmse_mean")
        for metric, title, name, color in [("rmse_mean", "Imputation RMSE Comparison", "error_comparison.png", PALETTE[0]), ("bias_mean", "Imputation Mean Bias", "bias_comparison.png", PALETTE[3]), ("runtime_seconds_mean", "Imputation Runtime", "runtime_comparison.png", PALETTE[2])]:
            fig, ax = plt.subplots(figsize=(9, 6)); ax.barh(subset.method, subset[metric], color=color); ax.set(title=f"{title} - {best_column}", xlabel=metric.replace("_", " ").title(), ylabel="Method"); paths.append(_finish(fig, output_dir / name))
        winners = benchmark.loc[benchmark["rank"].eq(1), "method"].value_counts().sort_values()
        fig, ax = plt.subplots(figsize=(8, 5)); ax.barh(winners.index, winners.values, color=PALETTE[4]); ax.set(title="Best-Method Frequency", xlabel="Columns won", ylabel="Method"); paths.append(_finish(fig, output_dir / "best_method_frequency.png"))
    return paths
