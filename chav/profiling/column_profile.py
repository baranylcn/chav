from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from chav.typing import ColumnType


@dataclass
class ColumnProfile:
    name: str
    dtype: ColumnType
    row_count: int = 0
    missing_count: int = 0
    missing_ratio: float = 0.0
    unique_count: int = 0
    uniqueness_ratio: float = 0.0
    dominant_value: Any = None
    dominant_value_share: float = 0.0
    mean: float | None = None
    std: float | None = None
    min_val: float | None = None
    max_val: float | None = None
    quantiles: dict[str, float] = field(default_factory=dict)
    cardinality: int | None = None
    parse_success_ratio: float | None = None

    @classmethod
    def from_series(cls, series: pd.Series, dtype: ColumnType) -> ColumnProfile:
        n = len(series)
        missing = int(series.isna().sum())
        non_null = series.dropna()
        nunique = non_null.nunique() if len(non_null) > 0 else 0

        dominant_val = None
        dominant_share = 0.0
        if len(non_null) > 0:
            vc = non_null.value_counts()
            dominant_val = vc.index[0]
            dominant_share = float(vc.iloc[0] / len(non_null))
            if hasattr(dominant_val, 'item'):
                dominant_val = dominant_val.item()

        prof = cls(
            name=series.name,
            dtype=dtype,
            row_count=n,
            missing_count=missing,
            missing_ratio=missing / n if n > 0 else 0.0,
            unique_count=nunique,
            uniqueness_ratio=nunique / len(non_null) if len(non_null) > 0 else 0.0,
            dominant_value=dominant_val,
            dominant_value_share=dominant_share,
        )

        if dtype == ColumnType.NUMERIC and len(non_null) > 0:
            numeric = pd.to_numeric(non_null, errors="coerce").dropna()
            if len(numeric) > 0:
                prof.mean = float(numeric.mean())
                prof.std = float(numeric.std()) if len(numeric) > 1 else 0.0
                prof.min_val = float(numeric.min())
                prof.max_val = float(numeric.max())
                prof.quantiles = {
                    "25%": float(numeric.quantile(0.25)),
                    "50%": float(numeric.quantile(0.50)),
                    "75%": float(numeric.quantile(0.75)),
                }

        if dtype == ColumnType.CATEGORICAL:
            prof.cardinality = nunique

        if dtype == ColumnType.DATETIME:
            parsed = pd.to_datetime(non_null, errors="coerce")
            prof.parse_success_ratio = float(parsed.notna().mean()) if len(non_null) > 0 else None

        return prof
