"""One-click governed report generation and exports."""
from __future__ import annotations
import os
import tkinter as tk
from tkinter import messagebox, ttk
from src.data_validator import DataValidator
from src.report_generator import generate_report
from src.recommendations import build_recommendations
from src.utils import ROOT, save_json
from .components import ScrollablePage, run_async


class ReportsPage(ScrollablePage):
    def __init__(self, parent: tk.Widget, app: object) -> None:
        super().__init__(parent, app, "Reports", "Generate executive, quality, imputation, model, and complete PDF deliverables")
        for label, command in [("Generate PDF Report", self.pdf), ("Generate Data-Quality Report", self.quality), ("Generate Imputation Report", self.imputation), ("Open Model Evaluation Report", self.model_report), ("Open Reports Folder", lambda: os.startfile(ROOT / "outputs/reports")), ("Export Executive Summary", self.summary)]: ttk.Button(self.content, text=label, style="Accent.TButton" if label.startswith("Generate PDF") else "TButton", command=command).pack(anchor="w", fill="x", pady=5, padx=8)
        self.message = ttk.Label(self.content, text="Reports include clear assumptions and limitations.", style="Subtitle.TLabel"); self.message.pack(anchor="w", pady=12)

    def pdf(self) -> None:
        if self.app.data is not None: run_async(self.app, lambda: generate_report(self.app.data, self.app.benchmark), lambda path: self.message.configure(text=f"Generated {path}"), "Generating PDF report")
    def quality(self) -> None:
        if self.app.data is not None: self.message.configure(text=f"Generated {DataValidator().export(DataValidator().validate(self.app.data))}")
    def imputation(self) -> None:
        payload = {"strategies": self.app.imputation_strategies, "comparison_rows": 0 if self.app.benchmark is None else len(self.app.benchmark)}; self.message.configure(text=f"Generated {save_json(payload, ROOT / 'outputs/reports/imputation_report.json')}")
    def model_report(self) -> None:
        path = ROOT / "outputs/reports/model_evaluation.json"; self.message.configure(text=f"Model evaluation: {path}"); os.startfile(path) if path.exists() else messagebox.showwarning("Model report", "Train the model first.")
    def summary(self) -> None:
        if self.app.data is not None: self.message.configure(text=f"Generated {save_json(build_recommendations(self.app.data, self.app.benchmark), ROOT / 'outputs/reports/executive_summary.json')}")
