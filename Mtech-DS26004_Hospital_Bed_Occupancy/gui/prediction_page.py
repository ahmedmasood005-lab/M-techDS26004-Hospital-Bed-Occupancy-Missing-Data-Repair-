"""Critical occupancy risk prediction form."""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
import joblib
import pandas as pd
from src.utils import ROOT, load_json
from .components import ScrollablePage
from .theme import COLORS


class PredictionPage(ScrollablePage):
    FIELDS = {"total_beds": 50, "occupied_beds": 42, "ICU_occupied": 8, "daily_admissions": 12, "daily_discharge": 8,
              "emergency_cases": 18, "staff_on_duty": 14, "department": "Emergency", "hospital_type": "Public",
              "weekend_flag": 0, "disease_outbreak_flag": 0}
    def __init__(self, parent: tk.Widget, app: object) -> None:
        super().__init__(parent, app, "ML Prediction", "Estimate critical occupancy probability and operational response")
        form = ttk.Frame(self.content, style="Card.TFrame", padding=18); form.pack(fill="x"); self.values = {}
        for i, (field, default) in enumerate(self.FIELDS.items()):
            ttk.Label(form, text=field.replace("_", " ").title(), style="Card.TLabel").grid(row=i//3*2, column=i%3, sticky="w", padx=8, pady=(5, 2)); variable = tk.StringVar(value=str(default)); self.values[field] = variable
            if field == "department": widget = ttk.Combobox(form, textvariable=variable, values=["Emergency", "ICU", "Cardiology", "Medicine", "Surgery", "Pediatrics"], state="readonly")
            elif field == "hospital_type": widget = ttk.Combobox(form, textvariable=variable, values=["Public", "Private", "Teaching"], state="readonly")
            elif field.endswith("flag"): widget = ttk.Combobox(form, textvariable=variable, values=["0", "1"], state="readonly")
            else: widget = ttk.Entry(form, textvariable=variable)
            widget.grid(row=i//3*2+1, column=i%3, sticky="ew", padx=8, pady=(0, 6)); form.columnconfigure(i%3, weight=1)
        ttk.Button(self.content, text="Predict Critical Risk", style="Accent.TButton", command=self.predict).pack(anchor="w", pady=12); self.result = tk.StringVar(value="Enter conditions and run prediction"); ttk.Label(self.content, textvariable=self.result, style="KPI.TLabel").pack(anchor="w"); self.gauge = ttk.Progressbar(self.content, maximum=100); self.gauge.pack(fill="x", pady=10)

    def predict(self) -> None:
        model_path = ROOT / "models/critical_occupancy_model.joblib"
        if not model_path.exists(): self.result.set("Train the model first."); return
        model = joblib.load(model_path); features = load_json(ROOT / "models/feature_names.json")
        base = self.app.repaired_data if self.app.repaired_data is not None else self.app.data
        if base is None: self.result.set("Load a dataset first."); return
        row = base.iloc[[0]].copy()
        for field, variable in self.values.items(): row[field] = variable.get() if field in {"department", "hospital_type"} else float(variable.get())
        row["occupancy_rate"] = row.occupied_beds / row.total_beds * 100; row["available_beds"] = (row.total_beds - row.occupied_beds - row.get("reserved_beds", 0)).clip(lower=0)
        probability = float(model.predict_proba(row[features])[:, 1][0]); percent = probability * 100; self.gauge["value"] = percent
        risk = "Low" if percent < 25 else "Moderate" if percent < 50 else "High" if percent < 75 else "Critical"
        recommendation = {"Low": "Continue routine monitoring.", "Moderate": "Review bed releases and next-shift staffing.", "High": "Activate capacity huddle and discharge coordination.", "Critical": "Escalate surge protocol and ICU/emergency capacity review now."}[risk]
        self.result.set(f"{risk} risk | Critical probability {percent:.1f}% | Confidence {max(percent, 100-percent):.1f}%\n{recommendation}\nKey factors: capacity, ICU load, admissions-discharge balance, emergency pressure, and staffing.")
