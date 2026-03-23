from __future__ import annotations

from scipy import stats as sp_stats

from chav.config import ChavConfig
from chav.profiling.compare_profile import CompareProfile
from chav.profiling.dataset_profile import DatasetProfile
from chav.rules.base import BaseRule
from chav.typing import ColumnType, Diagnostic, Severity, Status


class DriftRiskRule(BaseRule):
    name = "drift_risk"
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
        cfg = config.drift_risk
        psi_fail = cfg["psi_fail_threshold"]
        psi_warn = cfg["psi_warn_threshold"]
        ks_pvalue = cfg["ks_pvalue_threshold"]

        drifted = []
        evidence: dict = {}

        for col, cc in compare.columns.items():
            if cc.dtype == ColumnType.NUMERIC:
                psi = cc.numeric_drift_psi or 0.0
                ref_vals = compare.reference.df[col].dropna()
                cur_vals = compare.current.df[col].dropna()
                ks_stat, ks_p = 0.0, 1.0
                if len(ref_vals) > 1 and len(cur_vals) > 1:
                    ks_stat, ks_p = sp_stats.ks_2samp(ref_vals, cur_vals)

                if psi >= psi_fail or ks_p < ks_pvalue:
                    sev = "high"
                elif psi >= psi_warn:
                    sev = "medium"
                else:
                    continue

                drifted.append(col)
                evidence[col] = {
                    "psi": round(psi, 4),
                    "ks_statistic": round(ks_stat, 4),
                    "ks_pvalue": round(ks_p, 6),
                    "severity": sev,
                }

            elif cc.dtype == ColumnType.CATEGORICAL:
                tvd = cc.categorical_drift_tvd or 0.0
                if tvd >= 0.3:
                    sev = "high"
                elif tvd >= 0.15:
                    sev = "medium"
                else:
                    continue

                drifted.append(col)
                evidence[col] = {
                    "tvd": round(tvd, 4),
                    "severity": sev,
                }

        if not drifted:
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
            affected_columns=drifted,
            evidence=evidence,
        )
