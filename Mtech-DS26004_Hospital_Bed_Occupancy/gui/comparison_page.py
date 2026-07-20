"""Sortable repeated-mask method comparison with background execution."""
from __future__ import annotations
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from src.imputation_benchmark import benchmark
from src.imputation_methods import ImputationEngine
from .components import ScrollablePage, dataframe_tree, run_async


class ComparisonPage(ScrollablePage):
    def __init__(self, parent: tk.Widget, app: object) -> None:
        super().__init__(parent, app, "Compare Methods", "Bias-aware ranking across repeated artificial masks")
        bar = ttk.Frame(self.content); bar.pack(fill="x")
        ttk.Button(bar, text="Run Comparison", style="Accent.TButton", command=self.run).pack(side="left", padx=3); ttk.Button(bar, text="Stop Process", command=lambda: self.app.set_status("Stop requested; current atomic run will finish safely")).pack(side="left", padx=3); ttk.Button(bar, text="Export Comparison", command=self.export).pack(side="left", padx=3); ttk.Button(bar, text="Apply Recommended", command=self.apply).pack(side="left", padx=3); ttk.Button(bar, text="View Charts", command=lambda: self.app.show_page("Visual Analytics")).pack(side="left", padx=3)
        self.body = ttk.Frame(self.content); self.body.pack(fill="both", expand=True, pady=10)

    def run(self) -> None:
        if self.app.data is None: return
        columns = [c for c in ["occupancy_rate", "daily_discharge", "staff_on_duty", "infection_rate", "temperature", "ICU_occupied"] if c in self.app.data]
        methods = ["mean", "median", "forward_fill", "linear_interpolation", "knn", "iterative", "department_mean", "hospital_median", "rolling_median", "hybrid"]
        run_async(self.app, lambda: benchmark(self.app.data, columns=columns, methods=methods, seeds=(11, 29)), self._done, "Running repeated masking benchmark")

    def _done(self, result: pd.DataFrame) -> None:
        self.app.benchmark = result; winners = result[result["rank"].eq(1)]; self.app.best_imputer = winners.method.mode().iloc[0] if len(winners) else "Hybrid"; self.refresh()

    def refresh(self) -> None:
        for child in self.body.winfo_children(): child.destroy()
        if self.app.benchmark is None: ttk.Label(self.body, text="Run comparison or use the shipped experiment results.").pack(); return
        columns = [c for c in ["method", "column", "mae_mean", "rmse_mean", "bias_mean", "distribution_similarity_mean", "runtime_seconds_mean", "final_score", "rank"] if c in self.app.benchmark]
        tree = dataframe_tree(self.body, self.app.benchmark[columns], 16); tree.tag_configure("best", background="#D8F3E8"); tree.tag_configure("second", background="#DCEBFA"); tree.tag_configure("weak", background="#FFF0D2"); tree.tag_configure("bias", background="#FADDE1")
        for item in tree.get_children():
            values = tree.item(item, "values"); rank = float(values[-1]); bias = abs(float(values[4])); tree.item(item, tags=("best" if rank == 1 else "second" if rank == 2 else "bias" if bias > 2 else "weak" if rank >= 8 else "",))

    def export(self) -> None:
        if self.app.benchmark is None: return
        path = filedialog.asksaveasfilename(defaultextension=".csv");
        if path: self.app.benchmark.to_csv(path, index=False); messagebox.showinfo("Exported", path)

    def apply(self) -> None:
        if self.app.data is None or self.app.benchmark is None: return
        winners = self.app.benchmark[self.app.benchmark["rank"].eq(1)]; strategies = dict(zip(winners.column, winners.method)); repaired = ImputationEngine().repair(self.app.data, strategies); self.app.repaired_data = repaired.frame; self.app.imputation_strategies = repaired.strategy; self.app.set_status(f"Applied {len(repaired.strategy)} recommended strategies")
