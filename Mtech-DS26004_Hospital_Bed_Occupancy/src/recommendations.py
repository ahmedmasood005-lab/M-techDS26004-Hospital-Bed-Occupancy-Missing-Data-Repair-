"""Evidence-linked operational recommendations and limitations."""
from __future__ import annotations
from typing import Any
import pandas as pd


def build_recommendations(frame: pd.DataFrame, benchmark: pd.DataFrame | None = None) -> dict[str, Any]:
    """Separate computed findings, assumptions, actions, and limitations."""
    critical = float((frame.occupancy_rate >= 90).mean() * 100)
    missing = frame.isna().mean().sort_values(ascending=False)
    best = {}
    if benchmark is not None and not benchmark.empty:
        best = benchmark.loc[benchmark["rank"].eq(1)].set_index("column")["method"].to_dict()
    return {
        "verified_findings": [f"{critical:.1f}% of records meet the project critical threshold (>=90%).",
                              f"Highest missingness is {missing.index[0]} at {missing.iloc[0] * 100:.1f}%.",
                              f"Benchmark-backed column strategies are available for {len(best)} numerical columns."],
        "actions": ["Use department-level repair for staffing fields when repeated-mask validation confirms lower error.",
                    "Use temporal interpolation only for short gaps; route long sequential gaps and MNAR-designed fields to manual review.",
                    "Avoid mean imputation for skewed length-of-stay, mortality, and infection distributions.",
                    "Trigger capacity escalation at predicted critical risk >=42%, then review ICU, admissions, emergency load, and staffing.",
                    "Add mandatory-field validation, synchronization monitoring, and weekend staffing-data ownership at data entry."],
        "assumptions": ["The synthetic relationships approximate Pakistani multi-hospital operations but are not clinical ground truth.",
                        "The 90% critical threshold is a project convention and requires hospital-specific governance."],
        "limitations": ["Synthetic data may not represent every hospital.", "MNAR values cannot be reliably recovered from observed data alone.",
                        "Imputation cannot replace proper collection.", "KNN and iterative methods can be computationally expensive.",
                        "Interpolation can fail across long gaps.", "Model performance depends on data quality.",
                        "Critical thresholds vary by hospital, ward, and local policy."],
    }
