"""Missingness analytical findings and grouped breakdowns."""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
import pandas as pd
from src.missingness_analysis import grouped_missingness, mechanism_evidence, missingness_patterns, missingness_summary
from .components import ScrollablePage, dataframe_tree


class MissingDataPage(ScrollablePage):
    def __init__(self, parent: tk.Widget, app: object) -> None:
        super().__init__(parent, app, "Missing Data Analysis", "Column, department, hospital, date, weekday, season, pattern, and mechanism diagnostics")
        self.group = tk.StringVar(value="department"); chooser = ttk.Frame(self.content); chooser.pack(fill="x"); ttk.Combobox(chooser, textvariable=self.group, values=["department", "hospital_name", "date", "weekend_flag", "season"], state="readonly").pack(side="left"); ttk.Button(chooser, text="Analyze", style="Accent.TButton", command=self.refresh).pack(side="left", padx=6); self.body = ttk.Frame(self.content); self.body.pack(fill="both", expand=True, pady=8)

    def refresh(self) -> None:
        for child in self.body.winfo_children(): child.destroy()
        if self.app.data is None: ttk.Label(self.body, text="Load a dataset first.").pack(); return
        summary = missingness_summary(self.app.data).reset_index(names="column"); dataframe_tree(self.body, summary, 9)
        ttk.Label(self.body, text=f"Missingness by {self.group.get()}", style="KPI.TLabel").pack(anchor="w", pady=(12, 4)); grouped = grouped_missingness(self.app.data, self.group.get()).reset_index(); dataframe_tree(self.body, grouped, 7)
        ttk.Label(self.body, text="Common missingness patterns", style="KPI.TLabel").pack(anchor="w", pady=(12, 4)); dataframe_tree(self.body, missingness_patterns(self.app.data), 6)
        evidence = mechanism_evidence(self.app.data); text = tk.Text(self.body, height=10, wrap="word", relief="flat"); text.pack(fill="x", pady=8); text.insert("end", evidence["interpretation"] + "\n\n" + "\n".join(f"{c}: {v['likely_mechanism']}" for c, v in evidence["columns"].items())); text.configure(state="disabled")
