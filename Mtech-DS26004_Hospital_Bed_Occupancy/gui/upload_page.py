"""CSV import, preview, profiling, and sample loading."""
from __future__ import annotations
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from src.data_loader import load_dataset
from src.utils import ROOT
from .components import ScrollablePage, dataframe_tree


class UploadPage(ScrollablePage):
    def __init__(self, parent: tk.Widget, app: object) -> None:
        super().__init__(parent, app, "Upload Dataset", "Import CSV census data or load the generated demonstration dataset")
        actions = ttk.Frame(self.content); actions.pack(fill="x", pady=8)
        ttk.Button(actions, text="Select CSV", style="Accent.TButton", command=self.select).pack(side="left", padx=4); ttk.Button(actions, text="Load Sample Dataset", command=self.sample).pack(side="left", padx=4); ttk.Button(actions, text="Clear Dataset", command=self.clear).pack(side="left", padx=4)
        self.info = ttk.Label(self.content, text="Drop-style import zone: choose a CSV to begin", style="Subtitle.TLabel"); self.info.pack(fill="x", pady=10); self.preview = ttk.Frame(self.content); self.preview.pack(fill="both", expand=True)

    def select(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")]);
        if path: self.app.set_data(load_dataset(self.app.path_class(path)), self.app.path_class(path).name); self.refresh()

    def sample(self) -> None:
        path = ROOT / "data/raw/hospital_bed_occupancy_raw.csv"; self.app.set_data(load_dataset(path), path.name); self.refresh()

    def clear(self) -> None:
        if messagebox.askyesno("Clear dataset", "Remove the active dataset from this session?"): self.app.set_data(None, "No dataset") ; self.refresh()

    def refresh(self) -> None:
        for child in self.preview.winfo_children(): child.destroy()
        frame = self.app.data
        if frame is None: self.info.configure(text="No dataset loaded"); return
        self.info.configure(text=f"{self.app.dataset_name}  |  {len(frame):,} rows x {len(frame.columns)} columns  |  Missing: {frame.isna().sum().sum():,}  |  Duplicates: {frame.duplicated().sum():,}")
        dataframe_tree(self.preview, frame.head(100), 14)
