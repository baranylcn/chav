from __future__ import annotations

import pandas as pd

from chav.typing import ColumnType


def infer_column_type(series: pd.Series) -> ColumnType:
    if series.dropna().empty:
        return ColumnType.UNKNOWN

    dtype = series.dtype

    if pd.api.types.is_bool_dtype(dtype):
        return ColumnType.BOOLEAN

    if pd.api.types.is_datetime64_any_dtype(dtype):
        return ColumnType.DATETIME

    if pd.api.types.is_numeric_dtype(dtype):
        nunique = series.dropna().nunique()
        if nunique <= 2 and set(series.dropna().unique()).issubset({0, 1}):
            return ColumnType.BOOLEAN
        return ColumnType.NUMERIC

    if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype) or hasattr(dtype, "categories"):
        sample = series.dropna()
        if len(sample) > 0:
            try:
                pd.to_datetime(sample.head(50))
                sampled = pd.to_datetime(sample.sample(min(50, len(sample))), errors="coerce")
                if len(sample) <= 50 or sampled.notna().mean() > 0.8:
                    return ColumnType.DATETIME
            except (ValueError, TypeError):
                pass
        return ColumnType.CATEGORICAL

    return ColumnType.UNKNOWN


def infer_types(df: pd.DataFrame) -> dict[str, ColumnType]:
    return {col: infer_column_type(df[col]) for col in df.columns}
