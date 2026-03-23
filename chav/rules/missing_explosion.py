from __future__ import annotations

from chav.config import ChavConfig
from chav.profiling.compare_profile import CompareProfile
from chav.profiling.dataset_profile import DatasetProfile
from chav.rules.base import BaseRule
from chav.typing import Diagnostic, Severity, Status


class MissingExplosionRule(BaseRule):
    name = "missing_explosion"
    requires_reference = True

    def evaluate(
        self,
        profile: DatasetProfile,
        config: ChavConfig,
        compare: CompareProfile | None = None,
        target: str | None = None,
        time_column: str | None = None,
    ) -> Diagnostic:
        assert compare is not None
        cfg = config.missing_explosion
        abs_threshold = cfg["absolute_delta_threshold"]
        rel_threshold = cfg["relative_multiplier_threshold"]

        flagged = []
        evidence: dict = {}

        for col in compare.common_columns:
            ref_missing = compare.reference.columns[col].missing_ratio
            cur_missing = compare.current.columns[col].missing_ratio
            abs_delta = cur_missing - ref_missing
            rel_mult = (cur_missing / ref_missing) if ref_missing > 0 else (float("inf") if cur_missing > 0 else 1.0)

            if abs_delta >= abs_threshold or (rel_mult >= rel_threshold and abs_delta > 0.01):
                flagged.append(col)
                evidence[col] = {
                    "reference_missing_ratio": round(ref_missing, 4),
                    "current_missing_ratio": round(cur_missing, 4),
                    "absolute_delta": round(abs_delta, 4),
                    "relative_multiplier": round(rel_mult, 2) if rel_mult != float("inf") else "inf",
                }

        if not flagged:
            return Diagnostic(
                rule=self.name,
                status=Status.PASS,
                severity=Severity.LOW,
                confidence=1.0,
            )

        max_delta = max(
            compare.current.columns[c].missing_ratio - compare.reference.columns[c].missing_ratio for c in flagged
        )

        return Diagnostic(
            rule=self.name,
            status=Status.FAIL,
            severity=Severity.HIGH if max_delta > 0.3 else Severity.MEDIUM,
            confidence=min(0.7 + max_delta, 1.0),
            affected_columns=flagged,
            evidence=evidence,
        )
