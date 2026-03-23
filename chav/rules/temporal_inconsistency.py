from __future__ import annotations

from datetime import timedelta

import pandas as pd

from chav.config import ChavConfig
from chav.profiling.compare_profile import CompareProfile
from chav.profiling.dataset_profile import DatasetProfile
from chav.rules.base import BaseRule
from chav.typing import Diagnostic, Severity, Status


class TemporalInconsistencyRule(BaseRule):
    name = "temporal_inconsistency"
    requires_time_column = True

    def evaluate(
        self,
        profile: DatasetProfile,
        config: ChavConfig,
        compare: CompareProfile | None = None,
        target: str | None = None,
        time_column: str | None = None,
    ) -> Diagnostic:
        assert time_column is not None
        if time_column not in profile.df.columns:
            return self._skip()

        cfg = config.temporal_inconsistency
        future_tolerance = timedelta(hours=cfg["future_tolerance_hours"])
        ancient_floor = cfg["ancient_year_floor"]
        parse_fail_threshold = cfg["parse_failure_threshold"]

        raw = profile.df[time_column]
        parsed = pd.to_datetime(raw, errors="coerce")
        total = len(raw)
        parse_failures = int(parsed.isna().sum() - raw.isna().sum())
        parse_failure_ratio = max(parse_failures, 0) / total if total > 0 else 0.0

        valid = parsed.dropna()
        issue_count = 0
        evidence: dict = {
            "parse_failure_ratio": round(parse_failure_ratio, 4),
            "total_rows": total,
        }

        if parse_failure_ratio > parse_fail_threshold:
            issue_count += 1
            evidence["parse_failures"] = parse_failures

        if len(valid) > 0:
            now = pd.Timestamp.now()
            future_count = int((valid > now + future_tolerance).sum())
            evidence["future_timestamps"] = future_count
            evidence["future_ratio"] = round(future_count / len(valid), 4)
            if future_count > 0:
                issue_count += 1

            ancient_count = int((valid.dt.year < ancient_floor).sum())
            evidence["ancient_timestamps"] = ancient_count
            if ancient_count > 0:
                issue_count += 1

            dup_ts_ratio = float(valid.duplicated().mean())
            evidence["duplicate_timestamp_ratio"] = round(dup_ts_ratio, 4)
            if dup_ts_ratio > 0.5:
                issue_count += 1

        if issue_count == 0:
            return Diagnostic(
                rule=self.name,
                status=Status.PASS,
                severity=Severity.LOW,
                confidence=1.0,
                affected_columns=[time_column],
                evidence=evidence,
            )

        severity = Severity.HIGH if parse_failure_ratio > 0.1 else Severity.MEDIUM
        status = Status.FAIL if severity == Severity.HIGH else Status.WARN

        return Diagnostic(
            rule=self.name,
            status=status,
            severity=severity,
            confidence=min(0.6 + 0.1 * issue_count, 0.95),
            affected_columns=[time_column],
            evidence=evidence,
        )
