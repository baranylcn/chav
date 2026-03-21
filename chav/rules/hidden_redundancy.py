from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from chav.rules.base import BaseRule
from chav.typing import Diagnostic, Status, Severity, ColumnType
from chav.config import ChavConfig
from chav.profiling.dataset_profile import DatasetProfile
from chav.profiling.compare_profile import CompareProfile
from chav.utils.stats import cramers_v, correlation_ratio


class HiddenRedundancyRule(BaseRule):
    name = "hidden_redundancy"

    def evaluate(
        self,
        profile: DatasetProfile,
        config: ChavConfig,
        compare: CompareProfile | None = None,
        target: str | None = None,
        time_column: str | None = None,
    ) -> Diagnostic:
        cfg = config.hidden_redundancy
        corr_fail = cfg["correlation_fail_threshold"]
        corr_warn = cfg["correlation_warn_threshold"]
        cv_fail = cfg["cramers_v_fail_threshold"]
        cv_warn = cfg["cramers_v_warn_threshold"]
        eta_fail = cfg["eta_fail_threshold"]
        eta_warn = cfg["eta_warn_threshold"]

        df = profile.df
        types = profile.column_types

        numeric_cols = [c for c, t in types.items() if t == ColumnType.NUMERIC]
        categorical_cols = [c for c, t in types.items() if t == ColumnType.CATEGORICAL]

        redundant_pairs: list[dict] = []
        affected = set()

        for col_a, col_b in combinations(numeric_cols, 2):
            a_clean = pd.to_numeric(df[col_a], errors="coerce")
            b_clean = pd.to_numeric(df[col_b], errors="coerce")
            common = a_clean.dropna().index.intersection(b_clean.dropna().index)
            if len(common) < 10:
                continue

            r = float(np.corrcoef(a_clean.loc[common], b_clean.loc[common])[0, 1])
            abs_r = abs(r)

            if abs_r >= corr_warn:
                sev = "high" if abs_r >= corr_fail else "medium"
                redundant_pairs.append({
                    "columns": [col_a, col_b],
                    "method": "pearson",
                    "score": round(abs_r, 4),
                    "severity": sev,
                })
                affected.update([col_a, col_b])

        for col_a, col_b in combinations(categorical_cols, 2):
            if (profile.columns[col_a].cardinality or 0) > 100:
                continue
            if (profile.columns[col_b].cardinality or 0) > 100:
                continue

            cv = cramers_v(df[col_a], df[col_b])

            if cv >= cv_warn:
                sev = "high" if cv >= cv_fail else "medium"
                redundant_pairs.append({
                    "columns": [col_a, col_b],
                    "method": "cramers_v",
                    "score": round(cv, 4),
                    "severity": sev,
                })
                affected.update([col_a, col_b])

        for cat_col in categorical_cols:
            if (profile.columns[cat_col].cardinality or 0) > 50:
                continue
            for num_col in numeric_cols:
                eta = correlation_ratio(df[cat_col], pd.to_numeric(df[num_col], errors="coerce"))

                if eta >= eta_warn:
                    sev = "high" if eta >= eta_fail else "medium"
                    redundant_pairs.append({
                        "columns": [cat_col, num_col],
                        "method": "correlation_ratio",
                        "score": round(eta, 4),
                        "severity": sev,
                    })
                    affected.update([cat_col, num_col])

        if not redundant_pairs:
            return Diagnostic(
                rule=self.name,
                status=Status.PASS,
                severity=Severity.LOW,
                confidence=1.0,
            )

        has_high = any(p["severity"] == "high" for p in redundant_pairs)
        max_score = max(p["score"] for p in redundant_pairs)

        return Diagnostic(
            rule=self.name,
            status=Status.FAIL if has_high else Status.WARN,
            severity=Severity.HIGH if has_high else Severity.MEDIUM,
            confidence=round(min(max_score, 1.0), 4),
            affected_columns=sorted(affected),
            evidence={
                "pairs": sorted(redundant_pairs, key=lambda p: p["score"], reverse=True),
                "total_redundant_pairs": len(redundant_pairs),
            },
        )
