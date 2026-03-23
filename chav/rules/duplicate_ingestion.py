from __future__ import annotations

import pandas as pd

from chav.config import ChavConfig
from chav.profiling.compare_profile import CompareProfile
from chav.profiling.dataset_profile import DatasetProfile
from chav.rules.base import BaseRule
from chav.typing import Diagnostic, Severity, Status


class DuplicateIngestionRule(BaseRule):
    name = "duplicate_ingestion"

    def evaluate(
        self,
        profile: DatasetProfile,
        config: ChavConfig,
        compare: CompareProfile | None = None,
        target: str | None = None,
        time_column: str | None = None,
    ) -> Diagnostic:
        cfg = config.duplicate_ingestion
        fail_threshold = cfg["duplicate_ratio_threshold"]
        warn_threshold = cfg["warn_ratio_threshold"]

        df = profile.df
        dup_ratio = profile.duplicate_row_ratio
        dup_count = int(df.duplicated().sum())

        non_id_cols = [c for c, p in profile.columns.items() if p.uniqueness_ratio < 0.95]
        dup_ratio_no_id = 0.0
        if non_id_cols:
            dup_ratio_no_id = float(df[non_id_cols].duplicated().mean())

        evidence = {
            "duplicate_row_count": dup_count,
            "duplicate_ratio": round(dup_ratio, 4),
            "duplicate_ratio_excl_id_cols": round(dup_ratio_no_id, 4),
        }

        if time_column and time_column in df.columns:
            try:
                ts = pd.to_datetime(df[time_column], errors="coerce").dropna()
                if len(ts) > 0:
                    evidence["timestamp_duplicate_ratio"] = round(float(ts.duplicated().mean()), 4)
            except Exception:
                pass

        if dup_ratio >= fail_threshold:
            return Diagnostic(
                rule=self.name,
                status=Status.FAIL,
                severity=Severity.HIGH,
                confidence=min(0.7 + dup_ratio, 1.0),
                evidence=evidence,
            )
        elif dup_ratio >= warn_threshold:
            return Diagnostic(
                rule=self.name,
                status=Status.WARN,
                severity=Severity.MEDIUM,
                confidence=0.6,
                evidence=evidence,
            )
        else:
            return Diagnostic(
                rule=self.name,
                status=Status.PASS,
                severity=Severity.LOW,
                confidence=1.0,
                evidence=evidence,
            )
