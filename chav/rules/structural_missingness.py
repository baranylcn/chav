from __future__ import annotations

from itertools import combinations

import numpy as np

from chav.rules.base import BaseRule
from chav.typing import Diagnostic, Status, Severity
from chav.config import ChavConfig
from chav.profiling.dataset_profile import DatasetProfile
from chav.profiling.compare_profile import CompareProfile
from chav.utils.stats import phi_coefficient


class StructuralMissingnessRule(BaseRule):
    name = "structural_missingness"

    def evaluate(
        self,
        profile: DatasetProfile,
        config: ChavConfig,
        compare: CompareProfile | None = None,
        target: str | None = None,
        time_column: str | None = None,
    ) -> Diagnostic:
        cfg = config.structural_missingness
        phi_fail = cfg["phi_fail_threshold"]
        phi_warn = cfg["phi_warn_threshold"]
        min_null_ratio = cfg["min_null_ratio"]

        cols_with_nulls = [
            col for col, cp in profile.columns.items()
            if cp.missing_ratio >= min_null_ratio
        ]

        if len(cols_with_nulls) < 2:
            return Diagnostic(
                rule=self.name,
                status=Status.PASS,
                severity=Severity.LOW,
                confidence=1.0,
            )

        df = profile.df
        null_mask = {col: df[col].isna().values for col in cols_with_nulls}

        pairs: list[dict] = []
        affected = set()

        for col_a, col_b in combinations(cols_with_nulls, 2):
            phi = phi_coefficient(null_mask[col_a], null_mask[col_b])

            if phi >= phi_warn:
                co_null_count = int(np.sum(null_mask[col_a] & null_mask[col_b]))
                pairs.append({
                    "columns": [col_a, col_b],
                    "phi": round(phi, 4),
                    "co_null_count": co_null_count,
                    "co_null_ratio": round(co_null_count / profile.row_count, 4),
                    "severity": "high" if phi >= phi_fail else "medium",
                })
                affected.add(col_a)
                affected.add(col_b)

        if not pairs:
            return Diagnostic(
                rule=self.name,
                status=Status.PASS,
                severity=Severity.LOW,
                confidence=1.0,
            )

        has_high = any(p["severity"] == "high" for p in pairs)
        max_phi = max(p["phi"] for p in pairs)
        confidence = min(max_phi, 1.0)

        return Diagnostic(
            rule=self.name,
            status=Status.FAIL if has_high else Status.WARN,
            severity=Severity.HIGH if has_high else Severity.MEDIUM,
            confidence=round(confidence, 4),
            affected_columns=sorted(affected),
            evidence={
                "pairs": sorted(pairs, key=lambda p: p["phi"], reverse=True),
                "total_structural_pairs": len(pairs),
            },
        )
