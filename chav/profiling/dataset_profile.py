from __future__ import annotations

import pandas as pd

from chav.profiling.column_profile import ColumnProfile
from chav.typing import ColumnType
from chav.utils.type_inference import infer_types


class DatasetProfile:
    def __init__(self, df: pd.DataFrame, type_overrides: dict[str, ColumnType] | None = None):
        self.row_count = len(df)
        self.column_count = len(df.columns)
        self.duplicate_row_ratio = float(df.duplicated().mean()) if len(df) > 0 else 0.0
        self.total_missing_ratio = float(df.isna().mean().mean()) if df.size > 0 else 0.0

        inferred = infer_types(df)
        if type_overrides:
            inferred.update(type_overrides)
        self.column_types = inferred

        self.columns: dict[str, ColumnProfile] = {}
        for col in df.columns:
            self.columns[col] = ColumnProfile.from_series(df[col], inferred[col])

        self._df = df

    @property
    def df(self) -> pd.DataFrame:
        return self._df
