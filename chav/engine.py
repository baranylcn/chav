from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from chav.config import ChavConfig
from chav.profiling.dataset_profile import DatasetProfile
from chav.profiling.compare_profile import CompareProfile
from chav.rules import ALL_RULES
from chav.report import Report
from chav.typing import ColumnType


def analyze(
    data: pd.DataFrame | str | Path,
    reference_data: pd.DataFrame | str | Path | None = None,
    target: str | None = None,
    time_column: str | None = None,
    config: ChavConfig | None = None,
    type_overrides: dict[str, ColumnType] | None = None,
) -> Report:
    if config is None:
        config = ChavConfig()

    df = _load(data)
    ref_df = _load(reference_data) if reference_data is not None else None

    profile = DatasetProfile(df, type_overrides=type_overrides)

    compare = None
    if ref_df is not None:
        ref_profile = DatasetProfile(ref_df, type_overrides=type_overrides)
        compare = CompareProfile(ref_profile, profile)

    diagnostics = []
    for rule_cls in ALL_RULES:
        rule = rule_cls()
        diag = rule.safe_evaluate(
            profile=profile,
            config=config,
            compare=compare,
            target=target,
            time_column=time_column,
        )
        diagnostics.append(diag)

    return Report(
        diagnostics=diagnostics,
        rows=profile.row_count,
        columns=profile.column_count,
        has_reference=ref_df is not None,
        target=target,
        time_column=time_column,
    )


def _load(data: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    path = Path(data)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path.suffix}. Use CSV or pass a DataFrame.")
