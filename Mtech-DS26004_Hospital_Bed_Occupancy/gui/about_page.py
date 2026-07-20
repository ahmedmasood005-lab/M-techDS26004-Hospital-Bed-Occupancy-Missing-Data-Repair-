"""Project attribution, scope, tools, and disclaimer."""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from .components import ScrollablePage


class AboutPage(ScrollablePage):
    def __init__(self, parent: tk.Widget, app: object) -> None:
        super().__init__(parent, app, "About", "Academic identity, technical scope, version, and responsible-use statement")
        details = [("Project ID", "Mtech-DS26004"), ("Student", "Ahmed Masood"), ("Title", "Hospital Bed Occupancy Missing-Data Repair"), ("Version", "1.0.0"),
                   ("Tools", "Python, pandas, NumPy, SciPy, scikit-learn, XGBoost, Matplotlib, Seaborn, Plotly, SHAP, ReportLab, Tkinter"),
                   ("Description", "A benchmark-driven repair and critical-capacity decision-support prototype for irregular hospital census data."),
                   ("Disclaimer", "Synthetic educational data. Not a medical device and not approved for clinical decision-making. Human review and local validation are mandatory."),
                   ("GitHub", "https://github.com/your-username/Mtech-DS26004-Hospital-Bed-Occupancy")]
        card = ttk.Frame(self.content, style="Card.TFrame", padding=22); card.pack(fill="x")
        for row, (label, value) in enumerate(details): ttk.Label(card, text=label, style="KPI.TLabel", width=16).grid(row=row, column=0, sticky="nw", pady=7); ttk.Label(card, text=value, style="Card.TLabel", wraplength=700).grid(row=row, column=1, sticky="w", pady=7); card.columnconfigure(1, weight=1)
