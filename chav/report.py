from __future__ import annotations

import json
from typing import Any

import pandas as pd

from chav.typing import Diagnostic, Status
from chav.utils.formatting import format_summary

_ACTIONABLE = {Status.FAIL, Status.WARN, Status.ERROR}


class Report:
    def __init__(
        self,
        diagnostics: list[Diagnostic],
        rows: int,
        columns: int,
        has_reference: bool = False,
        target: str | None = None,
        time_column: str | None = None,
    ):
        self.diagnostics = diagnostics
        self.rows = rows
        self.columns = columns
        self.has_reference = has_reference
        self.target = target
        self.time_column = time_column

    @property
    def counts(self) -> dict[str, int]:
        c = {"pass": 0, "warn": 0, "fail": 0, "skipped": 0, "error": 0}
        for d in self.diagnostics:
            c[d.status.value] = c.get(d.status.value, 0) + 1
        return c

    def summary(self) -> str:
        return format_summary(self.diagnostics, self.rows, self.columns)

    def to_dict(self, all: bool = False) -> dict[str, Any]:
        diags = self.diagnostics if all else [d for d in self.diagnostics if d.status in _ACTIONABLE]

        return {
            "dataset_summary": {
                "rows": self.rows,
                "columns": self.columns,
            },
            "analysis_context": {
                "has_reference": self.has_reference,
                "target": self.target,
                "time_column": self.time_column,
            },
            "diagnostics": [d.to_dict() for d in diags],
            "counts": self.counts,
        }

    def to_json(self, indent: int = 2, all: bool = False) -> str:
        return json.dumps(self.to_dict(all=all), indent=indent, default=str)

    def to_dataframe(self, all: bool = False) -> pd.DataFrame:
        diags = self.diagnostics if all else [d for d in self.diagnostics if d.status in _ACTIONABLE]
        rows = [
            {
                "rule": d.rule,
                "status": d.status.value,
                "severity": d.severity.value,
                "confidence": round(d.confidence, 4),
                "affected_columns": ", ".join(d.affected_columns),
                "evidence": json.dumps(d.evidence, default=str),
            }
            for d in diags
        ]
        return pd.DataFrame(rows, columns=["rule", "status", "severity", "confidence", "affected_columns", "evidence"])

    def to_csv(self, path: str | None = None, all: bool = False) -> str | None:
        df = self.to_dataframe(all=all)
        if path is not None:
            df.to_csv(path, index=False)
            return None
        return df.to_csv(index=False)

    def to_excel(self, path: str, all: bool = False) -> None:
        try:
            import openpyxl  # noqa: F401
        except ImportError as exc:
            raise ImportError("openpyxl is required for to_excel(). Install it with: pip install openpyxl") from exc
        self.to_dataframe(all=all).to_excel(path, index=False)

    def __repr__(self) -> str:
        return self.to_json()

    def __str__(self) -> str:
        return self.to_json()
