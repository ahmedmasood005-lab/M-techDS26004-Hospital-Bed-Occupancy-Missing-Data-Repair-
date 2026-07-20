"""Responsive application shell, navigation, shared state, and lifecycle."""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any
import pandas as pd
from src.data_loader import load_dataset
from src.utils import ROOT, load_json
from .about_page import AboutPage
from .analytics_page import AnalyticsPage
from .comparison_page import ComparisonPage
from .dashboard_page import DashboardPage
from .imputation_page import ImputationPage
from .login_page import LoginPage
from .missing_data_page import MissingDataPage
from .prediction_page import PredictionPage
from .quality_page import QualityPage
from .repaired_data_page import RepairedDataPage
from .reports_page import ReportsPage
from .settings_page import SettingsPage
from .theme import COLORS, apply_theme
from .upload_page import UploadPage


class HospitalOccupancyApp(tk.Tk):
    """Main GUI controller with authenticated shell and non-blocking status feedback."""
    path_class = Path
    def __init__(self) -> None:
        super().__init__(); self.title("HarborBed Analytics | Mtech-DS26004"); self.geometry("1440x900"); self.minsize(1080, 700)
        self.data: pd.DataFrame | None = None; self.original_path: Path | None = None; self.dataset_name = "No dataset"; self.repaired_data = None; self.benchmark = None; self.imputation_strategies = {}; self.validation_report = {}; self.best_imputer = ""; self.model_metrics = {}
        settings = load_json(ROOT / "config/settings.json") if (ROOT / "config/settings.json").exists() else {}; apply_theme(self, settings.get("dark_mode", False), settings.get("font_size", 10))
        self.login = LoginPage(self, self.build_shell); self.login.pack(fill="both", expand=True)

    def build_shell(self) -> None:
        self.login.destroy(); self.sidebar = tk.Frame(self, bg=COLORS["navy"], width=225); self.sidebar.pack(side="left", fill="y"); self.sidebar.pack_propagate(False)
        brand = tk.Label(self.sidebar, text="H+  HarborBed", bg=COLORS["navy"], fg="white", font=("Segoe UI Semibold", 17), anchor="w", padx=18, pady=20); brand.pack(fill="x")
        self.main = ttk.Frame(self); self.main.pack(side="left", fill="both", expand=True); self.page_container = ttk.Frame(self.main); self.page_container.pack(fill="both", expand=True)
        classes = {"Dashboard": DashboardPage, "Upload Dataset": UploadPage, "Data Quality": QualityPage, "Missing Data Analysis": MissingDataPage,
                   "Imputation Lab": ImputationPage, "Compare Methods": ComparisonPage, "Repaired Dataset": RepairedDataPage, "ML Prediction": PredictionPage,
                   "Visual Analytics": AnalyticsPage, "Reports": ReportsPage, "Settings": SettingsPage, "About": AboutPage}
        self.pages = {name: cls(self.page_container, self) for name, cls in classes.items()}
        for page in self.pages.values(): page.place(relx=0, rely=0, relwidth=1, relheight=1)
        icons = ["▦", "⇧", "✓", "◉", "⚗", "≋", "▤", "◆", "◫", "▣", "⚙", "ⓘ"]
        for icon, name in zip(icons, classes): tk.Button(self.sidebar, text=f"{icon}   {name}", command=lambda n=name: self.show_page(n), bg=COLORS["navy"], fg="#DCE7F3", activebackground=COLORS["blue"], activeforeground="white", relief="flat", bd=0, anchor="w", padx=18, pady=7, font=("Segoe UI", 10), cursor="hand2").pack(fill="x")
        tk.Button(self.sidebar, text="↪   Logout", command=self.logout, bg=COLORS["navy"], fg="#FFB4B8", activebackground=COLORS["red"], activeforeground="white", relief="flat", bd=0, anchor="w", padx=18, pady=9).pack(side="bottom", fill="x")
        footer = ttk.Frame(self.main, style="Card.TFrame", padding=(12, 6)); footer.pack(fill="x", side="bottom"); self.status = tk.StringVar(value="Ready"); self.clock = tk.StringVar(); self.dataset_status = tk.StringVar(value=self.dataset_name)
        ttk.Label(footer, textvariable=self.status, style="Card.TLabel").pack(side="left"); self.progress = ttk.Progressbar(footer, mode="indeterminate", length=130); self.progress.pack(side="left", padx=12); ttk.Label(footer, textvariable=self.dataset_status, style="Card.TLabel").pack(side="right", padx=12); ttk.Label(footer, textvariable=self.clock, style="Card.TLabel").pack(side="right"); self.tick()
        self._load_shipped_state(); self.show_page("Dashboard")

    def _load_shipped_state(self) -> None:
        raw = ROOT / "data/raw/hospital_bed_occupancy_raw.csv"
        if raw.exists(): self.set_data(load_dataset(raw), raw.name); self.original_path = raw
        comparison = ROOT / "outputs/experiments/imputation_comparison.csv"
        if comparison.exists(): self.benchmark = pd.read_csv(comparison); winners = self.benchmark[self.benchmark["rank"].eq(1)]; self.best_imputer = winners.method.mode().iloc[0] if len(winners) else ""
        evaluation = ROOT / "outputs/reports/model_evaluation.json"
        if evaluation.exists(): self.model_metrics = load_json(evaluation).get("test_metrics", {})

    def show_page(self, name: str) -> None:
        page = self.pages[name]; page.tkraise(); page.refresh(); self.set_status(f"Viewing {name}")

    def set_data(self, frame: pd.DataFrame | None, name: str) -> None:
        self.data, self.dataset_name = frame, name; self.dataset_status.set(name) if hasattr(self, "dataset_status") else None; self.repaired_data = None; self.refresh_pages()

    def reload_active_dataset(self) -> None:
        path = self.original_path or ROOT / "data/raw/hospital_bed_occupancy_raw.csv"
        if path.exists(): self.set_data(load_dataset(path), path.name)

    def refresh_pages(self) -> None:
        if hasattr(self, "pages"):
            for page in self.pages.values():
                try: page.refresh()
                except Exception: continue

    def set_status(self, text: str) -> None:
        if hasattr(self, "status"): self.status.set(text)

    def show_error(self, text: str) -> None:
        self.set_status("Operation failed"); messagebox.showerror("HarborBed Analytics", text)

    def apply_preferences(self, settings: dict[str, Any]) -> None:
        apply_theme(self, settings.get("dark_mode", False), settings.get("font_size", 10))

    def tick(self) -> None:
        if hasattr(self, "clock"): self.clock.set(datetime.now().strftime("%d %b %Y  %H:%M:%S"))
        self.after(1000, self.tick)

    def logout(self) -> None:
        if messagebox.askyesno("Logout", "End this authenticated session?"):
            for widget in self.winfo_children(): widget.destroy()
            self.login = LoginPage(self, self.build_shell); self.login.pack(fill="both", expand=True)
