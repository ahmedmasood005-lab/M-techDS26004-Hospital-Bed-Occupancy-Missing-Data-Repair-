"""Filtered visual analytics and PNG export launcher."""
from __future__ import annotations
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from src.eda import create_charts
from src.utils import ROOT
from .components import ScrollablePage, run_async


class AnalyticsPage(ScrollablePage):
    def __init__(self, parent: tk.Widget, app: object) -> None:
        super().__init__(parent, app, "Visual Analytics", "Filter hospital operations and export publication-ready PNG evidence")
        self.hospital, self.department, self.season = tk.StringVar(value="All"), tk.StringVar(value="All"), tk.StringVar(value="All")
        bar = ttk.Frame(self.content); bar.pack(fill="x"); self.boxes = []
        for variable, label in [(self.hospital, "Hospital"), (self.department, "Department"), (self.season, "Season")]: ttk.Label(bar, text=label).pack(side="left", padx=(8, 2)); box = ttk.Combobox(bar, textvariable=variable, state="readonly", width=20); box.pack(side="left"); self.boxes.append(box)
        ttk.Button(bar, text="Generate Charts", style="Accent.TButton", command=self.generate).pack(side="left", padx=8); ttk.Button(bar, text="Open Charts Folder", command=lambda: os.startfile(ROOT / "outputs/charts")).pack(side="left")
        self.status = ttk.Label(self.content, text="Charts are saved at 180 DPI.", style="Subtitle.TLabel"); self.status.pack(anchor="w", pady=12)

    def refresh(self) -> None:
        if self.app.data is not None:
            self.boxes[0]["values"] = ["All", *sorted(self.app.data.hospital_name.dropna().unique())]; self.boxes[1]["values"] = ["All", *sorted(self.app.data.department.dropna().unique())]; self.boxes[2]["values"] = ["All", *sorted(self.app.data.season.dropna().unique())]

    def generate(self) -> None:
        if self.app.data is None: return
        data = self.app.data.copy()
        for column, variable in [("hospital_name", self.hospital), ("department", self.department), ("season", self.season)]:
            if variable.get() != "All": data = data[data[column].eq(variable.get())]
        run_async(self.app, lambda: create_charts(data, self.app.repaired_data, self.app.benchmark), lambda paths: self.status.configure(text=f"Generated {len(paths)} charts in {ROOT / 'outputs/charts'}"), "Generating analytical charts")
