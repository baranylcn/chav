from __future__ import annotations

import re

from chav.config import ChavConfig
from chav.profiling.compare_profile import CompareProfile
from chav.profiling.dataset_profile import DatasetProfile
from chav.rules.base import BaseRule
from chav.typing import ColumnType, Diagnostic, Severity, Status

_ID_PATTERNS = re.compile(r"(^id$|_id$|^id_|uuid|hash|token|_key$|^key_)", re.IGNORECASE)


class IdLikeFeaturesRule(BaseRule):
    name = "id_like_features"

    def evaluate(
        self,
        profile: DatasetProfile,
        config: ChavConfig,
        compare: CompareProfile | None = None,
        target: str | None = None,
        time_column: str | None = None,
    ) -> Diagnostic:
        cfg = config.id_like_features
        threshold = cfg["uniqueness_threshold"]
        min_rows = cfg["min_rows"]

        if profile.row_count < min_rows:
            return self._skip()

        flagged = []
        evidence: dict = {}

        for col_name, col_prof in profile.columns.items():
            if col_prof.dtype in (ColumnType.DATETIME, ColumnType.NUMERIC, ColumnType.BOOLEAN):
                continue

            signals = 0
            reasons = []

            if col_prof.uniqueness_ratio >= threshold:
                signals += 2
                reasons.append(f"uniqueness={col_prof.uniqueness_ratio:.3f}")

            if _ID_PATTERNS.search(col_name):
                signals += 1
                reasons.append("name_match")

            if col_prof.dtype == ColumnType.CATEGORICAL and col_prof.unique_count > 0:
                sample = profile.df[col_name].dropna().head(20).astype(str)
                lengths = sample.str.len()
                if lengths.std() <= 1.0 and len(sample) >= 5:
                    signals += 1
                    reasons.append("consistent_lengths")

            if signals >= 2:
                flagged.append(col_name)
                evidence[col_name] = {
                    "uniqueness_ratio": round(col_prof.uniqueness_ratio, 4),
                    "unique_count": col_prof.unique_count,
                    "signals": reasons,
                }

        if not flagged:
            return Diagnostic(
                rule=self.name,
                status=Status.PASS,
                severity=Severity.LOW,
                confidence=1.0,
            )

        return Diagnostic(
            rule=self.name,
            status=Status.WARN,
            severity=Severity.MEDIUM,
            confidence=min(0.6 + 0.1 * len(flagged), 0.95),
            affected_columns=flagged,
            evidence=evidence,
        )
