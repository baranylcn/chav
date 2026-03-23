from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class DatasetSummary(BaseModel):
    rows: int
    columns: int


class AnalysisContext(BaseModel):
    has_reference: bool
    target: str | None
    time_column: str | None


class DiagnosticOut(BaseModel):
    rule: str
    status: str
    severity: str
    confidence: float
    affected_columns: list[str]
    evidence: dict[str, Any]


class ReportOut(BaseModel):
    dataset_summary: DatasetSummary
    analysis_context: AnalysisContext
    diagnostics: list[DiagnosticOut]
    counts: dict[str, int]
