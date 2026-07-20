"""Executive home dashboard."""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
import pandas as pd
from .components import KPI, ScrollablePage
from .theme import COLORS


class DashboardPage(ScrollablePage):
    def __init__(self, parent: tk.Widget, app: object) -> None:
        super().__init__(parent, app, "Command Center", "Live census quality, capacity, and critical-risk overview")
        self.cards = ttk.Frame(self.content); self.cards.pack(fill="x"); self.chart_area = ttk.Frame(self.content, style="Card.TFrame", padding=16); self.chart_area.pack(fill="both", expand=True, pady=14)

    def refresh(self) -> None:
        for child in self.cards.winfo_children(): child.destroy()
        frame = self.app.data
        if frame is None: ttk.Label(self.cards, text="Load the sample dataset to activate the dashboard.").pack(pady=30); return
        values = [("Records", f"{len(frame):,}", COLORS["blue"]), ("Hospitals", str(frame.hospital_name.nunique()), COLORS["cyan"]),
                  ("Departments", str(frame.department.nunique()), COLORS["purple"]), ("Total beds", f"{frame.total_beds.sum():,.0f}", COLORS["blue"]),
                  ("Occupied", f"{frame.occupied_beds.sum():,.0f}", COLORS["orange"]), ("Available", f"{frame.available_beds.sum():,.0f}", COLORS["green"]),
                  ("Avg occupancy", f"{frame.occupancy_rate.mean():.1f}%", COLORS["orange"]), ("Missing cells", f"{frame.isna().sum().sum():,}", COLORS["red"]),
                  ("Missing data", f"{frame.isna().sum().sum()/frame.size*100:.1f}%", COLORS["purple"]), ("Critical cases", f"{(frame.occupancy_rate>=90).sum():,}", COLORS["red"]),
                  ("Best ROC-AUC", f"{self.app.model_metrics.get('roc_auc', 0):.3f}", COLORS["green"]), ("Best imputer", self.app.best_imputer or "Run benchmark", COLORS["cyan"])]
        for i, item in enumerate(values): KPI(self.cards, *item).grid(row=i//4, column=i%4, sticky="ew", padx=5, pady=5); self.cards.columnconfigure(i%4, weight=1)
        for child in self.chart_area.winfo_children(): child.destroy()
        summary = frame.groupby("department").occupancy_rate.mean().sort_values(ascending=False)
        ttk.Label(self.chart_area, text="Department capacity pulse", style="KPI.TLabel").pack(anchor="w")
        for name, value in summary.items():
            row = ttk.Frame(self.chart_area, style="Card.TFrame"); row.pack(fill="x", pady=5); ttk.Label(row, text=name, style="Card.TLabel", width=15).pack(side="left")
            bar = ttk.Progressbar(row, maximum=100, value=value); bar.pack(side="left", fill="x", expand=True, padx=8); ttk.Label(row, text=f"{value:.1f}%", style="Card.TLabel", width=8).pack(side="right")
