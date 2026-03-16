from __future__ import annotations

import numpy as np
import pandas as pd


def compute_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref_clean = reference.dropna()
    cur_clean = current.dropna()
    if len(ref_clean) < 2 or len(cur_clean) < 2:
        return 0.0

    breakpoints = np.percentile(ref_clean, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0

    ref_counts = np.histogram(ref_clean, bins=breakpoints)[0]
    cur_counts = np.histogram(cur_clean, bins=breakpoints)[0]

    ref_pct = (ref_counts + 1) / (ref_counts.sum() + len(ref_counts))
    cur_pct = (cur_counts + 1) / (cur_counts.sum() + len(cur_counts))

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def categorical_drift_score(reference: pd.Series, current: pd.Series) -> float:
    ref_dist = reference.value_counts(normalize=True)
    cur_dist = current.value_counts(normalize=True)
    all_cats = set(ref_dist.index) | set(cur_dist.index)
    tvd = 0.0
    for cat in all_cats:
        tvd += abs(ref_dist.get(cat, 0.0) - cur_dist.get(cat, 0.0))
    return float(tvd / 2)
