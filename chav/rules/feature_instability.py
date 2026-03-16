from __future__ import annotations

from chav.rules.base import BaseRule
from chav.typing import Diagnostic, Status, Severity, ColumnType
from chav.config import ChavConfig
from chav.profiling.dataset_profile import DatasetProfile
from chav.profiling.compare_profile import CompareProfile


class FeatureInstabilityRule(BaseRule):
    name = "feature_instability"
    requires_reference = True

    def evaluate(
        self,
        profile: DatasetProfile,
        config: ChavConfig,
        compare: CompareProfile | None = None,
        target: str | None = None,
        time_column: str | None = None,
    ) -> Diagnostic:
        cfg = config.feature_instability
        fail_threshold = cfg["instability_fail_threshold"]
        warn_threshold = cfg["instability_warn_threshold"]

        unstable = []
        evidence: dict = {}

        for col, cc in compare.columns.items():
            factors = []
            score = 0.0

            miss_delta = abs(cc.missing_delta)
            miss_contrib = min(miss_delta * 2, 1.0)
            if miss_delta > 0.01:
                factors.append(f"missingness_delta={cc.missing_delta:+.3f}")
            score += miss_contrib * 0.4

            if cc.dtype == ColumnType.NUMERIC and cc.numeric_drift_psi is not None:
                drift_contrib = min(cc.numeric_drift_psi / 0.5, 1.0)
                if cc.numeric_drift_psi > 0.05:
                    factors.append(f"psi={cc.numeric_drift_psi:.3f}")
                score += drift_contrib * 0.3
            elif cc.dtype == ColumnType.CATEGORICAL and cc.categorical_drift_tvd is not None:
                drift_contrib = min(cc.categorical_drift_tvd / 0.5, 1.0)
                if cc.categorical_drift_tvd > 0.05:
                    factors.append(f"tvd={cc.categorical_drift_tvd:.3f}")
                score += drift_contrib * 0.3

            if cc.dtype == ColumnType.CATEGORICAL and cc.cardinality_growth_factor is not None:
                growth = cc.cardinality_growth_factor
                card_contrib = min(abs(growth - 1.0), 1.0)
                if abs(growth - 1.0) > 0.1:
                    factors.append(f"cardinality_growth={growth:.2f}x")
                score += card_contrib * 0.3

            if cc.dtype == ColumnType.NUMERIC and cc.variance_change_ratio is not None:
                var_ratio = cc.variance_change_ratio
                if var_ratio != float('inf'):
                    var_contrib = min(abs(var_ratio - 1.0), 1.0)
                    if abs(var_ratio - 1.0) > 0.2:
                        factors.append(f"variance_ratio={var_ratio:.2f}")
                    score += var_contrib * 0.3

            if score >= warn_threshold and factors:
                unstable.append(col)
                evidence[col] = {
                    "instability_score": round(score, 4),
                    "factors": factors,
                }

        if not unstable:
            return Diagnostic(
                rule=self.name,
                status=Status.PASS,
                severity=Severity.LOW,
                confidence=1.0,
            )

        max_score = max(evidence[c]["instability_score"] for c in unstable)

        return Diagnostic(
            rule=self.name,
            status=Status.FAIL if max_score >= fail_threshold else Status.WARN,
            severity=Severity.HIGH if max_score >= fail_threshold else Severity.MEDIUM,
            confidence=min(0.6 + max_score * 0.3, 0.95),
            affected_columns=unstable,
            evidence=evidence,
        )
