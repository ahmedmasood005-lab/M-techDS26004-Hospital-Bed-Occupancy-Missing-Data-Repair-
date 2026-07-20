"""Generate the polished project PDF with evidence, tables, charts, and caveats."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, PageBreak, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)

from .recommendations import build_recommendations
from .utils import ROOT, load_json

NAVY, BLUE, CYAN, LIGHT, ORANGE = colors.HexColor("#102A43"), colors.HexColor("#176BCE"), colors.HexColor("#15B7C9"), colors.HexColor("#F2F6FA"), colors.HexColor("#F59E0B")


def _header_footer(canvas: Any, doc: Any) -> None:
    canvas.setTitle("Mtech-DS26004 Hospital Bed Occupancy Missing-Data Repair")
    canvas.setAuthor("Ahmed Masood")
    canvas.setSubject("Advanced missing-data imputation benchmark and critical occupancy model")
    canvas.saveState(); canvas.setFillColor(NAVY); canvas.rect(0, A4[1] - 18 * mm, A4[0], 18 * mm, fill=1, stroke=0)
    canvas.setFillColor(colors.white); canvas.setFont("Helvetica-Bold", 9); canvas.drawString(18 * mm, A4[1] - 11 * mm, "Mtech-DS26004 | Hospital Bed Occupancy Missing-Data Repair")
    canvas.setFillColor(NAVY); canvas.setFont("Helvetica", 8); canvas.drawString(18 * mm, 10 * mm, "Ahmed Masood | Academic portfolio project")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}"); canvas.restoreState()


def generate_report(frame: pd.DataFrame, benchmark: pd.DataFrame | None = None, output_path: Path | None = None) -> Path:
    """Build a comprehensive project report from generated evidence."""
    output_path = output_path or ROOT / "outputs/reports/Mtech_DS26004_Project_Report.pdf"; output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet(); styles.add(ParagraphStyle(name="Cover", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=28, leading=34, textColor=colors.white, alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading1"], fontSize=18, leading=22, textColor=NAVY, spaceBefore=10, spaceAfter=8))
    styles.add(ParagraphStyle(name="Sub", parent=styles["Heading2"], fontSize=13, textColor=BLUE, spaceBefore=8, spaceAfter=5))
    doc = BaseDocTemplate(str(output_path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=24 * mm, bottomMargin=18 * mm)
    doc.addPageTemplates(PageTemplate(id="main", frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="content")], onPage=_header_footer))
    story: list[Any] = [Spacer(1, 32 * mm), Table([[Paragraph("HOSPITAL BED OCCUPANCY<br/>MISSING-DATA REPAIR", styles["Cover"])]], colWidths=[174 * mm], rowHeights=[76 * mm], style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), NAVY), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")])), Spacer(1, 12 * mm),
                        Paragraph("Advanced Data Science Project", styles["Title"]), Paragraph("Project ID: Mtech-DS26004", styles["Heading2"]), Paragraph("Student: Ahmed Masood", styles["Heading2"]), Spacer(1, 8 * mm), Paragraph("Version 1.0 | Python 3.10+ | Tkinter | scikit-learn", styles["BodyText"]), PageBreak()]
    missing_pct = frame.isna().sum().sum() / frame.size * 100; critical_pct = (frame.occupancy_rate >= 90).mean() * 100
    sections = [
        ("1. Project Information", "This portfolio project delivers a reproducible data-repair benchmark, leakage-aware critical-occupancy classifier, interactive administrator desktop application, and governed export workflow."),
        ("2. Executive Summary", f"The synthetic multi-hospital census contains {len(frame):,} daily ward records. Targeted fields contain {missing_pct:.2f}% missing cells overall, generated through MCAR, MAR, MNAR, sequential, departmental, weekend, holiday, and block mechanisms. {critical_pct:.1f}% of observed records meet the project critical threshold."),
        ("3. Problem Statement and Objectives", "Delayed entry, sensor failure, incomplete discharge records, synchronization errors, and human mistakes can distort capacity decisions. The objective is to measure repair error and bias through repeated artificial masking, select a defensible method per numerical column, and predict critical occupancy without target leakage."),
        ("4. Stakeholders", "Hospital administrators need capacity alerts; ward managers need operational detail; data stewards need missingness diagnostics; analysts need reproducible comparisons; clinicians require transparent limitations and human oversight."),
        ("5. Dataset Description", "Five hospitals, six departments, 180 daily census dates, operational capacity, ICU, flow, staffing, outcomes, climate, calendar, and outbreak fields. Constraints ensure occupied beds do not exceed total beds, ICU occupancy does not exceed ICU capacity, and bed balances remain non-negative."),
        ("6. Data Quality Findings", "The reusable validator checks identifiers, dates, duplicates, negative values, categorical domains, bed and ICU limits, staffing consistency, balance equations, occupancy bounds, and robust extreme outliers. Missing values are preserved until benchmarked; impossible values are conservatively converted or bounded."),
        ("7. Missing-Data Patterns", "The generator records known mechanisms. Chi-square associations between missing indicators and department, hospital, season, weekend, and holiday provide MAR-like evidence. Mortality and length of stay include MNAR-designed masking; observed data alone cannot prove or fully recover MNAR values."),
        ("8. Methodology", "Known values are repeatedly hidden using deterministic seeds. Each method imputes the masked values, which are then compared with their ground truth. Per-column scoring combines RMSE, MAE, absolute bias, KS distance, variance distortion, correlation preservation, and runtime."),
        ("9. Statistical Analysis", "Descriptive statistics characterize central tendency and shape. Shapiro-Wilk tests normality on a bounded sample; Welch t-test and Mann-Whitney compare weekend occupancy; ANOVA and Kruskal-Wallis compare hospitals and departments; chi-square tests categorical association; KS compares distributions; confidence intervals quantify uncertainty."),
        ("10. Imputation Methods", "The engine implements mean, median, mode, forward fill, backward fill, linear, time, and polynomial interpolation, KNN, iterative chained equations, department mean, hospital median, seasonal median, rolling median, and a hybrid data-aware strategy."),
    ]
    for title, text in sections: story.extend([Paragraph(title, styles["Section"]), Paragraph(text, styles["BodyText"]), Spacer(1, 4 * mm)])
    if benchmark is not None and not benchmark.empty:
        story.extend([Paragraph("11. Benchmark Results and Best Method per Column", styles["Section"]), Paragraph("Rank 1 is selected independently for each column. Lower error and bias increase the score; stronger distribution and correlation preservation increase the score; runtime is a smaller operational penalty.", styles["BodyText"])])
        winners = benchmark[benchmark["rank"].eq(1)].head(18)
        rows = [["Column", "Method", "MAE", "RMSE", "Bias", "Score"]] + [[str(r.column), str(r.method), f"{r.mae_mean:.3f}", f"{r.rmse_mean:.3f}", f"{r.bias_mean:.3f}", f"{r.final_score:.1f}"] for _, r in winners.iterrows()]
        table = Table(rows, repeatRows=1, colWidths=[39 * mm, 31 * mm, 23 * mm, 23 * mm, 23 * mm, 22 * mm]); table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), .3, colors.HexColor("#CBD5E1")), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 7), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        story.extend([Spacer(1, 5 * mm), table])
    more = [
        ("12. Bias Analysis", "A method can achieve low average error while systematically over- or under-estimating capacity. Mean bias, percentage bias, variance and standard-deviation distortion, KS statistic, Wasserstein distance, and correlation preservation are therefore reported with point-error metrics."),
        ("13. Feature Engineering", "Operational ratios, pressure indices, staffing gaps, shortage and risk flags are supplemented by lags and rolling means. All temporal features shift before rolling or use expanding past-only aggregates to prevent look-ahead leakage."),
        ("14. Machine Learning Methodology", "Seven classifier families are supported through scikit-learn pipelines and a column transformer. ID, date, occupied beds, available beds, occupancy rate, and threshold-derived flags are excluded from predictors. Stratified train, validation, and test partitions are fixed by random seed."),
        ("15. Model Evaluation", "Accuracy, precision, recall, specificity, F1, ROC-AUC, PR-AUC, balanced accuracy, MCC, log loss, Brier score, calibration, confusion matrix, ROC, and precision-recall curves are saved. Selection weights recall at 60% and ROC-AUC at 40%, reflecting the cost of missed critical events while monitoring false alarms."),
        ("16. Ethical Considerations", "Outputs support operational review and must not be treated as clinical decisions. Protected attributes are absent; synthetic records contain no real patient identities. Human review is required for MNAR repair, long gaps, and high-risk alerts. Local governance should set thresholds and monitor subgroup error."),
    ]
    for title, text in more: story.extend([Paragraph(title, styles["Section"]), Paragraph(text, styles["BodyText"]), Spacer(1, 4 * mm)])
    for chart_name, caption in [("occupancy_trend.png", "Average occupancy over time with the project threshold."), ("missing_percentage.png", "Targeted missingness percentages by field."), ("best_method_frequency.png", "Frequency of per-column benchmark winners."), ("roc_curve.png", "Out-of-sample discrimination of the selected model."), ("shap_summary.png", "Mean absolute SHAP importance for the selected model.")]:
        chart = ROOT / "outputs/charts" / chart_name
        if chart.exists(): story.extend([Paragraph(caption, styles["Sub"]), Image(str(chart), width=160 * mm, height=90 * mm), Spacer(1, 4 * mm)])
    recommendations = build_recommendations(frame, benchmark)
    for title, key in [("17. Business Recommendations", "actions"), ("18. Assumptions", "assumptions"), ("19. Limitations", "limitations")]:
        story.append(Paragraph(title, styles["Section"])); story.extend([Paragraph(f"- {item}", styles["BodyText"]) for item in recommendations[key]])
    story.extend([Paragraph("20. Future Improvements", styles["Section"]), Paragraph("Validate on governed real-world hospital data; add temporal cross-validation, drift monitoring, uncertainty intervals, cost-sensitive thresholds, role-based access, database-backed audit trails, and hospital-specific calibration.", styles["BodyText"]),
                  Paragraph("21. Conclusion", styles["Section"]), Paragraph("The project demonstrates that missing-data repair should be selected per field through ground-truth masking and bias-aware metrics, not by a single default. The integrated benchmark, model, GUI, tests, and report form a reproducible decision-support prototype.", styles["BodyText"]),
                  Paragraph("22. References", styles["Section"]), Paragraph("scikit-learn documentation; pandas documentation; SciPy statistical reference; Little and Rubin, Statistical Analysis with Missing Data; ReportLab User Guide; project source code and generated artifacts.", styles["BodyText"])])
    doc.build(story); return output_path
