"""Repaired dataset preview and governed export."""
from __future__ import annotations
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from src.imputation_methods import ImputationEngine
from .components import ScrollablePage, dataframe_tree


class RepairedDataPage(ScrollablePage):
    def __init__(self, parent: tk.Widget, app: object) -> None:
        super().__init__(parent, app, "Repaired Dataset", "Review before/after completeness, applied strategies, and portable exports")
        bar = ttk.Frame(self.content); bar.pack(fill="x")
        ttk.Button(bar, text="Create Hybrid Repair", style="Accent.TButton", command=self.repair).pack(side="left", padx=3); ttk.Button(bar, text="Export CSV", command=lambda: self.export("csv")).pack(side="left", padx=3); ttk.Button(bar, text="Export Excel", command=lambda: self.export("xlsx")).pack(side="left", padx=3); ttk.Button(bar, text="Reset Changes", command=self.reset).pack(side="left", padx=3)
        self.summary = ttk.Label(self.content, text="No repaired dataset yet", style="Subtitle.TLabel"); self.summary.pack(anchor="w", pady=10); self.body = ttk.Frame(self.content); self.body.pack(fill="both", expand=True)

    def repair(self) -> None:
        if self.app.data is None: return
        result = ImputationEngine().repair(self.app.data); self.app.repaired_data, self.app.imputation_strategies = result.frame, result.strategy; self.refresh()

    def refresh(self) -> None:
        for child in self.body.winfo_children(): child.destroy()
        if self.app.repaired_data is None: return
        before = self.app.data.isna().sum().sum() if self.app.data is not None else 0; after = self.app.repaired_data.isna().sum().sum(); self.summary.configure(text=f"Missing cells: {before:,} before → {after:,} after | {len(self.app.imputation_strategies)} column strategies")
        strategies = pd.DataFrame(self.app.imputation_strategies.items(), columns=["column", "applied_strategy"]); dataframe_tree(self.body, strategies, 7); dataframe_tree(self.body, self.app.repaired_data.head(120), 10)

    def export(self, kind: str) -> None:
        if self.app.repaired_data is None: return
        path = filedialog.asksaveasfilename(defaultextension=f".{kind}");
        if path:
            if kind == "csv": self.app.repaired_data.to_csv(path, index=False)
            else: self.app.repaired_data.to_excel(path, index=False)
            messagebox.showinfo("Exported", path)

    def reset(self) -> None:
        self.app.repaired_data = None; self.app.imputation_strategies = {}; self.summary.configure(text="Changes reset"); self.refresh()
