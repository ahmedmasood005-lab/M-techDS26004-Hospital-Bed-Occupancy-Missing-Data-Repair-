"""Validation dashboard and report export."""
from __future__ import annotations
import tkinter as tk
from tkinter import messagebox, ttk
import pandas as pd
from src.data_validator import DataValidator
from src.data_cleaner import clean_data
from .components import KPI, ScrollablePage, dataframe_tree
from .theme import COLORS


class QualityPage(ScrollablePage):
    def __init__(self, parent: tk.Widget, app: object) -> None:
        super().__init__(parent, app, "Data Quality", "Rule-based validation, outlier review, quality scoring, and export")
        actions = ttk.Frame(self.content); actions.pack(fill="x", pady=5); ttk.Button(actions, text="Run Validation", style="Accent.TButton", command=self.refresh).pack(side="left", padx=3); ttk.Button(actions, text="Apply Safe Corrections", command=self.correct).pack(side="left", padx=3); self.body = ttk.Frame(self.content); self.body.pack(fill="both", expand=True)

    def correct(self) -> None:
        if self.app.data is not None and messagebox.askyesno("Safe corrections", "Deduplicate, normalize labels, bound invalid capacity values, and preserve missing cells for repair?"):
            self.app.set_data(clean_data(self.app.data), self.app.dataset_name); self.refresh()

    def refresh(self) -> None:
        for child in self.body.winfo_children(): child.destroy()
        if self.app.data is None: ttk.Label(self.body, text="Load a dataset first.").pack(); return
        validator = DataValidator(); report = validator.validate(self.app.data); self.app.validation_report = report
        cards = ttk.Frame(self.body); cards.pack(fill="x")
        for i, item in enumerate([("Quality score", f"{report['quality_score']:.1f}/100", COLORS["green"]), ("Missing", f"{report['missing_cells']:,}", COLORS["orange"]), ("Duplicates", str(report['duplicate_rows']), COLORS["blue"]), ("Issues", str(sum(x['count'] for x in report['issues'])), COLORS["red"])]): KPI(cards, *item).grid(row=0, column=i, sticky="ew", padx=5); cards.columnconfigure(i, weight=1)
        issues = pd.DataFrame(report["issues"] or [{"rule": "none", "severity": "pass", "count": 0, "description": "All validation rules passed"}]); dataframe_tree(self.body, issues, 10)
        ttk.Button(self.body, text="Export Validation Report", command=lambda: messagebox.showinfo("Exported", str(validator.export(report)))).pack(anchor="e", pady=8)
