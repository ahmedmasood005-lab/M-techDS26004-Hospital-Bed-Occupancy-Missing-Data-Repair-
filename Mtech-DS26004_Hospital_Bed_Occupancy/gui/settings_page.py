"""Persisted visual and export preferences."""
from __future__ import annotations
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from src.utils import ROOT, save_json
from .components import ScrollablePage


class SettingsPage(ScrollablePage):
    def __init__(self, parent: tk.Widget, app: object) -> None:
        super().__init__(parent, app, "Settings", "Theme, typography, chart style, and export preferences")
        self.dark = tk.BooleanVar(value=False); self.font = tk.IntVar(value=10); self.chart = tk.StringVar(value="whitegrid"); self.export = tk.StringVar(value=str(ROOT / "outputs"))
        form = ttk.Frame(self.content, style="Card.TFrame", padding=20); form.pack(fill="x")
        ttk.Checkbutton(form, text="Dark mode", variable=self.dark).grid(row=0, column=0, sticky="w", pady=6); ttk.Label(form, text="Font size", style="Card.TLabel").grid(row=1, column=0, sticky="w"); ttk.Spinbox(form, from_=9, to=16, textvariable=self.font).grid(row=1, column=1, sticky="ew")
        ttk.Label(form, text="Chart style", style="Card.TLabel").grid(row=2, column=0, sticky="w"); ttk.Combobox(form, textvariable=self.chart, values=["whitegrid", "darkgrid", "ticks"], state="readonly").grid(row=2, column=1, sticky="ew")
        ttk.Label(form, text="Default export folder", style="Card.TLabel").grid(row=3, column=0, sticky="w"); ttk.Entry(form, textvariable=self.export).grid(row=3, column=1, sticky="ew"); ttk.Button(form, text="Browse", command=self.browse).grid(row=3, column=2, padx=5); form.columnconfigure(1, weight=1)
        ttk.Button(self.content, text="Save & Apply", style="Accent.TButton", command=self.save).pack(anchor="w", pady=10); ttk.Button(self.content, text="Reset Settings", command=self.reset).pack(anchor="w")
    def browse(self) -> None:
        path = filedialog.askdirectory(); self.export.set(path or self.export.get())
    def save(self) -> None:
        payload = {"dark_mode": self.dark.get(), "font_size": self.font.get(), "chart_style": self.chart.get(), "default_export_folder": self.export.get(), "application_theme": "Harbor Analytics"}; save_json(payload, ROOT / "config/settings.json"); self.app.apply_preferences(payload); messagebox.showinfo("Settings", "Preferences saved and applied.")
    def reset(self) -> None:
        self.dark.set(False); self.font.set(10); self.chart.set("whitegrid"); self.export.set(str(ROOT / "outputs")); self.save()
