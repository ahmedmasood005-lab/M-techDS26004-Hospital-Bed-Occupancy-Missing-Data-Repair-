"""Purpose-labeled descriptive and inferential hospital statistics."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from scipy import stats


def run_statistical_tests(frame: pd.DataFrame) -> dict[str, Any]:
    """Run normality, group-difference, association, and confidence-interval tests."""
    occupancy = frame.occupancy_rate.dropna()
    sample = occupancy.sample(min(5000, len(occupancy)), random_state=42)
    weekend = frame.loc[frame.weekend_flag.eq(1), "occupancy_rate"].dropna()
    weekday = frame.loc[frame.weekend_flag.eq(0), "occupancy_rate"].dropna()
    hospital_groups = [group.occupancy_rate.dropna().to_numpy() for _, group in frame.groupby("hospital_name")]
    department_groups = [group.occupancy_rate.dropna().to_numpy() for _, group in frame.groupby("department")]
    contingency = pd.crosstab(frame.department, frame.disease_outbreak_flag)
    mean, sem = occupancy.mean(), stats.sem(occupancy)
    ci = stats.t.interval(.95, len(occupancy) - 1, loc=mean, scale=sem)
    return {
        "descriptive": {"mean": mean, "median": occupancy.median(), "mode": occupancy.mode().iloc[0],
                        "variance": occupancy.var(), "standard_deviation": occupancy.std(), "skewness": occupancy.skew(), "kurtosis": occupancy.kurt()},
        "shapiro_wilk": {"purpose": "Assess normality on a bounded sample", "statistic": stats.shapiro(sample).statistic, "p_value": stats.shapiro(sample).pvalue},
        "weekend_t_test": {"purpose": "Compare weekend and weekday mean occupancy", "statistic": stats.ttest_ind(weekend, weekday, equal_var=False).statistic, "p_value": stats.ttest_ind(weekend, weekday, equal_var=False).pvalue},
        "mann_whitney": {"purpose": "Robust non-normal weekend comparison", "statistic": stats.mannwhitneyu(weekend, weekday).statistic, "p_value": stats.mannwhitneyu(weekend, weekday).pvalue},
        "hospital_anova": {"purpose": "Test mean differences across hospitals", "statistic": stats.f_oneway(*hospital_groups).statistic, "p_value": stats.f_oneway(*hospital_groups).pvalue},
        "department_kruskal": {"purpose": "Test distribution differences across departments", "statistic": stats.kruskal(*department_groups).statistic, "p_value": stats.kruskal(*department_groups).pvalue},
        "chi_square": {"purpose": "Test department/outbreak categorical association", "statistic": stats.chi2_contingency(contingency).statistic, "p_value": stats.chi2_contingency(contingency).pvalue},
        "confidence_interval_95": [ci[0], ci[1]],
        "pearson_occupied_admissions": frame[["occupied_beds", "daily_admissions"]].corr(method="pearson").iloc[0, 1],
        "spearman_staff_occupancy": frame[["staff_on_duty", "occupancy_rate"]].corr(method="spearman").iloc[0, 1],
        "ks_weekend_weekday": {"purpose": "Compare weekend and weekday distributions", "statistic": stats.ks_2samp(weekend, weekday).statistic, "p_value": stats.ks_2samp(weekend, weekday).pvalue},
        "mcar_note": "Little's test is not run because mixed-type high-dimensional data and designed MAR/MNAR mechanisms violate its practical assumptions; missing indicators are tested against observed covariates instead.",
    }
