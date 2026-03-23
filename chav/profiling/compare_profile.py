from __future__ import annotations

from dataclasses import dataclass, field

from chav.profiling.dataset_profile import DatasetProfile
from chav.typing import ColumnType
from chav.utils.stats import categorical_drift_score, compute_psi


@dataclass
class ColumnCompare:
    name: str
    dtype: ColumnType
    missing_delta: float = 0.0
    cardinality_delta: int = 0
    cardinality_growth_factor: float = 1.0
    unseen_category_ratio: float = 0.0
    unseen_categories: list[str] = field(default_factory=list)
    numeric_drift_psi: float | None = None
    categorical_drift_tvd: float | None = None
    mean_delta: float | None = None
    std_delta: float | None = None
    variance_change_ratio: float | None = None


class CompareProfile:
    def __init__(self, reference: DatasetProfile, current: DatasetProfile):
        self.reference = reference
        self.current = current
        self.common_columns = [c for c in reference.df.columns if c in current.df.columns]
        self.columns: dict[str, ColumnCompare] = {}

        for col in self.common_columns:
            ref_prof = reference.columns[col]
            cur_prof = current.columns[col]
            dtype = reference.column_types.get(col, ColumnType.UNKNOWN)

            cc = ColumnCompare(name=col, dtype=dtype)
            cc.missing_delta = cur_prof.missing_ratio - ref_prof.missing_ratio

            if dtype == ColumnType.CATEGORICAL:
                ref_card = ref_prof.cardinality or 0
                cur_card = cur_prof.cardinality or 0
                cc.cardinality_delta = cur_card - ref_card
                if ref_card > 0:
                    cc.cardinality_growth_factor = cur_card / ref_card
                elif cur_card > 0:
                    cc.cardinality_growth_factor = float("inf")
                else:
                    cc.cardinality_growth_factor = 1.0

                ref_vals = set(reference.df[col].dropna().unique())
                cur_vals = set(current.df[col].dropna().unique())
                unseen = cur_vals - ref_vals
                cc.unseen_category_ratio = len(unseen) / len(cur_vals) if cur_vals else 0.0
                cc.unseen_categories = [str(v) for v in list(unseen)[:10]]

                cc.categorical_drift_tvd = categorical_drift_score(reference.df[col].dropna(), current.df[col].dropna())

            if dtype == ColumnType.NUMERIC:
                cc.numeric_drift_psi = compute_psi(reference.df[col], current.df[col])
                if ref_prof.mean is not None and cur_prof.mean is not None:
                    cc.mean_delta = cur_prof.mean - ref_prof.mean
                if ref_prof.std is not None and cur_prof.std is not None:
                    cc.std_delta = cur_prof.std - ref_prof.std
                    if ref_prof.std > 0:
                        cc.variance_change_ratio = cur_prof.std / ref_prof.std
                    else:
                        cc.variance_change_ratio = float("inf") if cur_prof.std > 0 else 1.0

            self.columns[col] = cc
