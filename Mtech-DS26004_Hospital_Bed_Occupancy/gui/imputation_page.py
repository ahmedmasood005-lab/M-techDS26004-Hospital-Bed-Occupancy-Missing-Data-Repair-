"""Interactive imputation laboratory with undo and distribution metrics."""
from __future__ import annotations
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from src.imputation_methods import ImputationEngine, METHODS
from .components import ScrollablePage, dataframe_tree


class ImputationPage(ScrollablePage):
    def __init__(self, parent: tk.Widget, app: object) -> None:
        super().__init__(parent, app, "Imputation Lab", "Configure, preview, apply, compare, undo, reset, and export individual repairs")
        self.column, self.method, self.parameter, self.metrics = tk.StringVar(), tk.StringVar(value="median"), tk.StringVar(value="5"), tk.StringVar(value="Select a missing column")
        bar = ttk.Frame(self.content); bar.pack(fill="x"); self.column_box = ttk.Combobox(bar, textvariable=self.column, state="readonly", width=26); self.column_box.pack(side="left", padx=4); ttk.Combobox(bar, textvariable=self.method, values=METHODS, state="readonly", width=24).pack(side="left", padx=4)
        ttk.Label(bar, text="Parameter").pack(side="left", padx=(8, 2)); ttk.Entry(bar, textvariable=self.parameter, width=7).pack(side="left")
        for label, command in [("Apply", self.apply), ("Undo", self.undo), ("Reset", self.reset), ("Save CSV", self.save)]: ttk.Button(bar, text=label, style="Accent.TButton" if label == "Apply" else "TButton", command=command).pack(side="left", padx=3)
        ttk.Label(self.content, textvariable=self.metrics, style="Subtitle.TLabel").pack(anchor="w", pady=9); self.preview = ttk.Frame(self.content); self.preview.pack(fill="both", expand=True); self.history: list[pd.DataFrame] = []

    def refresh(self) -> None:
        if self.app.data is not None:
            columns = self.app.data.columns[self.app.data.isna().any()].tolist(); self.column_box.configure(values=columns)
            if columns and self.column.get() not in columns: self.column.set(columns[0])

    def apply(self) -> None:
        if self.app.data is None or not self.column.get(): return
        self.history.append(self.app.data.copy()); before = self.app.data[self.column.get()].copy()
        try: value = int(self.parameter.get())
        except ValueError: value = 5
        filled = ImputationEngine().apply(self.app.data, self.column.get(), self.method.get(), n_neighbors=value, window=value, order=max(1, min(value, 3))); self.app.data[self.column.get()] = filled
        missing = before.isna(); ks = ks_2samp(before.dropna(), filled).statistic; self.metrics.set(f"Filled {missing.sum():,} cells | mean shift {filled.mean()-before.mean():.4f} | KS distribution difference {ks:.4f}")
        comparison = pd.DataFrame({"before": before[missing].head(100), "after": filled[missing].head(100)}); self._preview(comparison); self.app.refresh_pages()

    def _preview(self, frame: pd.DataFrame) -> None:
        for child in self.preview.winfo_children(): child.destroy(); dataframe_tree(self.preview, frame.reset_index(), 12)

    def undo(self) -> None:
        if self.history: self.app.data = self.history.pop(); self.metrics.set("Last imputation undone"); self.app.refresh_pages()

    def reset(self) -> None:
        self.app.reload_active_dataset(); self.history.clear(); self.metrics.set("Dataset reset from disk")

    def save(self) -> None:
        if self.app.data is None: return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")]);
        if path: self.app.data.to_csv(path, index=False); messagebox.showinfo("Saved", path)
