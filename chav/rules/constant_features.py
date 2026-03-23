from __future__ import annotations

from chav.config import ChavConfig
from chav.profiling.compare_profile import CompareProfile
from chav.profiling.dataset_profile import DatasetProfile
from chav.rules.base import BaseRule
from chav.typing import Diagnostic, Severity, Status


class ConstantFeaturesRule(BaseRule):
    name = "constant_features"

    def evaluate(
        self,
        profile: DatasetProfile,
        config: ChavConfig,
        compare: CompareProfile | None = None,
        target: str | None = None,
        time_column: str | None = None,
    ) -> Diagnostic:
        cfg = config.constant_features
        near_threshold = cfg["near_constant_threshold"]

        constant_cols = []
        near_constant_cols = []
        evidence: dict = {}

        for col_name, col_prof in profile.columns.items():
            if col_prof.unique_count <= 1:
                constant_cols.append(col_name)
                evidence[col_name] = {
                    "unique_count": col_prof.unique_count,
                    "dominant_value": col_prof.dominant_value,
                    "dominant_share": round(col_prof.dominant_value_share, 4),
                    "type": "constant",
                }
            elif col_prof.dominant_value_share >= near_threshold:
                near_constant_cols.append(col_name)
                evidence[col_name] = {
                    "unique_count": col_prof.unique_count,
                    "dominant_value": col_prof.dominant_value,
                    "dominant_share": round(col_prof.dominant_value_share, 4),
                    "type": "near_constant",
                }

        all_flagged = constant_cols + near_constant_cols

        if not all_flagged:
            return Diagnostic(
                rule=self.name,
                status=Status.PASS,
                severity=Severity.LOW,
                confidence=1.0,
            )

        return Diagnostic(
            rule=self.name,
            status=Status.FAIL if constant_cols else Status.WARN,
            severity=Severity.HIGH if constant_cols else Severity.MEDIUM,
            confidence=1.0 if constant_cols else 0.8,
            affected_columns=all_flagged,
            evidence=evidence,
        )
