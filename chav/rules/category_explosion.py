from __future__ import annotations

from chav.config import ChavConfig
from chav.profiling.compare_profile import CompareProfile
from chav.profiling.dataset_profile import DatasetProfile
from chav.rules.base import BaseRule
from chav.typing import ColumnType, Diagnostic, Severity, Status


class CategoryExplosionRule(BaseRule):
    name = "category_explosion"
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
        cfg = config.category_explosion
        growth_fail = cfg["cardinality_growth_threshold"]
        unseen_fail = cfg["unseen_ratio_threshold"]
        growth_warn = cfg["warn_growth_threshold"]

        flagged = []
        evidence: dict = {}

        for col, cc in compare.columns.items():
            if cc.dtype != ColumnType.CATEGORICAL:
                continue

            ref_card = compare.reference.columns[col].cardinality or 0
            cur_card = compare.current.columns[col].cardinality or 0
            if ref_card == 0:
                continue

            growth = cc.cardinality_growth_factor
            unseen = cc.unseen_category_ratio

            if growth >= growth_fail or unseen >= unseen_fail:
                sev = "high"
            elif growth >= growth_warn:
                sev = "medium"
            else:
                continue

            flagged.append(col)
            evidence[col] = {
                "reference_cardinality": ref_card,
                "current_cardinality": cur_card,
                "growth_factor": round(growth, 2),
                "unseen_ratio": round(unseen, 4),
                "sample_new_values": cc.unseen_categories[:5],
                "severity": sev,
            }

        if not flagged:
            return Diagnostic(
                rule=self.name,
                status=Status.PASS,
                severity=Severity.LOW,
                confidence=1.0,
            )

        has_high = any(e.get("severity") == "high" for e in evidence.values())

        return Diagnostic(
            rule=self.name,
            status=Status.FAIL if has_high else Status.WARN,
            severity=Severity.HIGH if has_high else Severity.MEDIUM,
            confidence=0.85,
            affected_columns=flagged,
            evidence=evidence,
        )
