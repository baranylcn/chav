from __future__ import annotations

import numpy as np
import pandas as pd

from chav.rules.base import BaseRule
from chav.typing import Diagnostic, Status, Severity, ColumnType
from chav.config import ChavConfig
from chav.profiling.dataset_profile import DatasetProfile
from chav.profiling.compare_profile import CompareProfile
from chav.utils.stats import compute_psi


class ConditionalDriftRule(BaseRule):
    name = "conditional_drift"
    requires_reference = True

    def evaluate(
        self,
        profile: DatasetProfile,
        config: ChavConfig,
        compare: CompareProfile | None = None,
        target: str | None = None,
        time_column: str | None = None,
    ) -> Diagnostic:
        cfg = config.conditional_drift
        max_card = cfg["max_stratify_cardinality"]
        min_subgroup = cfg["min_subgroup_size"]
        psi_fail = cfg["psi_fail_threshold"]
        psi_warn = cfg["psi_warn_threshold"]
        disproportion = cfg["disproportion_factor"]

        ref_df = compare.reference.df
        cur_df = compare.current.df
        ref_types = compare.reference.column_types

        stratify_cols = [
            col for col in compare.common_columns
            if ref_types.get(col) == ColumnType.CATEGORICAL
            and 2 <= (compare.reference.columns[col].cardinality or 0) <= max_card
        ]

        numeric_cols = [
            col for col in compare.common_columns
            if ref_types.get(col) == ColumnType.NUMERIC
        ]

        if not stratify_cols or not numeric_cols:
            return Diagnostic(
                rule=self.name,
                status=Status.PASS,
                severity=Severity.LOW,
                confidence=1.0,
            )

        findings: list[dict] = []
        affected = set()

        for strat_col in stratify_cols:
            ref_groups = ref_df.groupby(strat_col)
            cur_groups = cur_df.groupby(strat_col)
            common_groups = set(ref_groups.groups.keys()) & set(cur_groups.groups.keys())

            for num_col in numeric_cols:
                cc = compare.columns.get(num_col)
                overall_psi = 0.0
                if cc and cc.numeric_drift_psi is not None:
                    overall_psi = cc.numeric_drift_psi

                subgroup_results = []
                for group_val in common_groups:
                    ref_sub = ref_groups.get_group(group_val)[num_col].dropna()
                    cur_sub = cur_groups.get_group(group_val)[num_col].dropna()

                    if len(ref_sub) < min_subgroup or len(cur_sub) < min_subgroup:
                        continue

                    sub_psi = compute_psi(ref_sub, cur_sub)
                    subgroup_results.append({
                        "group_value": str(group_val),
                        "subgroup_psi": round(sub_psi, 4),
                        "ref_size": len(ref_sub),
                        "cur_size": len(cur_sub),
                    })

                flagged = _flag_subgroups(
                    subgroup_results, overall_psi,
                    psi_warn, psi_fail, disproportion,
                )

                if flagged:
                    max_sub_psi = max(s["subgroup_psi"] for s in flagged)
                    findings.append({
                        "stratify_column": strat_col,
                        "numeric_column": num_col,
                        "overall_psi": round(overall_psi, 4),
                        "subgroups": sorted(flagged, key=lambda s: s["subgroup_psi"], reverse=True),
                        "severity": "high" if max_sub_psi >= psi_fail else "medium",
                    })
                    affected.update([strat_col, num_col])

        if not findings:
            return Diagnostic(
                rule=self.name,
                status=Status.PASS,
                severity=Severity.LOW,
                confidence=1.0,
            )

        has_high = any(f["severity"] == "high" for f in findings)
        max_psi = max(
            s["subgroup_psi"]
            for f in findings
            for s in f["subgroups"]
        )
        confidence = round(min(max_psi / psi_fail, 1.0), 4)

        return Diagnostic(
            rule=self.name,
            status=Status.FAIL if has_high else Status.WARN,
            severity=Severity.HIGH if has_high else Severity.MEDIUM,
            confidence=confidence,
            affected_columns=sorted(affected),
            evidence={
                "findings": sorted(
                    findings,
                    key=lambda f: max(s["subgroup_psi"] for s in f["subgroups"]),
                    reverse=True,
                ),
                "total_conditional_drifts": len(findings),
            },
        )


def _flag_subgroups(
    subgroup_results: list[dict],
    overall_psi: float,
    psi_warn: float,
    psi_fail: float,
    disproportion: float,
) -> list[dict]:
    flagged = []
    for s in subgroup_results:
        sub_psi = s["subgroup_psi"]
        if sub_psi < psi_warn:
            continue

        reason = None
        if overall_psi < psi_warn:
            reason = "hidden"
        elif overall_psi > 0 and sub_psi / overall_psi >= disproportion:
            reason = "disproportionate"

        if reason:
            flagged.append({**s, "drift_type": reason})

    return flagged
