from __future__ import annotations

import numpy as np
import pandas as pd


def compute_psi(reference: pd.Series, current: pd.Series, bins: int | None = None) -> float:
    ref_clean = reference.dropna()
    cur_clean = current.dropna()
    if len(ref_clean) < 2 or len(cur_clean) < 2:
        return 0.0

    if bins is None:
        bins = _adaptive_bins(len(ref_clean))

    breakpoints = np.percentile(ref_clean, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0

    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts = np.histogram(ref_clean, bins=breakpoints)[0]
    cur_counts = np.histogram(cur_clean, bins=breakpoints)[0]

    ref_pct = (ref_counts + 1) / (ref_counts.sum() + len(ref_counts))
    cur_pct = (cur_counts + 1) / (cur_counts.sum() + len(cur_counts))

    psi: float = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def _adaptive_bins(n: int) -> int:
    if n < 30:
        return 3
    if n < 100:
        return 5
    if n < 500:
        return 8
    return 10


def categorical_drift_score(reference: pd.Series, current: pd.Series) -> float:
    ref_dist = reference.value_counts(normalize=True)
    cur_dist = current.value_counts(normalize=True)
    all_cats = set(ref_dist.index) | set(cur_dist.index)
    tvd = 0.0
    for cat in all_cats:
        tvd += abs(ref_dist.get(cat, 0.0) - cur_dist.get(cat, 0.0))
    return float(tvd / 2)


def phi_coefficient(a_null: np.ndarray, b_null: np.ndarray) -> float:
    n = len(a_null)
    if n == 0:
        return 0.0

    n11 = int(np.sum(a_null & b_null))
    n00 = int(np.sum(~a_null & ~b_null))
    n10 = int(np.sum(a_null & ~b_null))
    n01 = int(np.sum(~a_null & b_null))

    row1 = n11 + n10
    row0 = n01 + n00
    col1 = n11 + n01
    col0 = n10 + n00

    denom = np.sqrt(float(row1) * row0 * col1 * col0)
    if denom == 0:
        return 0.0

    return float((n11 * n00 - n10 * n01) / denom)


def cramers_v(col_a: pd.Series, col_b: pd.Series) -> float:
    from scipy.stats import chi2_contingency

    a_clean = col_a.dropna()
    b_clean = col_b.dropna()
    common_idx = a_clean.index.intersection(b_clean.index)
    if len(common_idx) < 2:
        return 0.0

    ct = pd.crosstab(col_a.loc[common_idx], col_b.loc[common_idx])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return 0.0

    chi2, _, _, _ = chi2_contingency(ct)
    n = ct.values.sum()
    k = min(ct.shape[0], ct.shape[1])
    if k <= 1 or n == 0:
        return 0.0

    return float(np.sqrt(chi2 / (n * (k - 1))))


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    common_idx = categories.dropna().index.intersection(values.dropna().index)
    if len(common_idx) < 2:
        return 0.0

    cats = categories.loc[common_idx]
    vals = values.loc[common_idx].astype(float)

    grand_mean = vals.mean()
    groups = vals.groupby(cats)

    ss_between = 0.0
    for _, group_vals in groups:
        ss_between += len(group_vals) * (group_vals.mean() - grand_mean) ** 2

    ss_total: float = np.sum((vals - grand_mean) ** 2)
    if ss_total == 0:
        return 0.0

    return float(np.sqrt(ss_between / ss_total))
