from __future__ import annotations

import re

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from chav.config import ChavConfig
from chav.profiling.compare_profile import CompareProfile
from chav.profiling.dataset_profile import DatasetProfile
from chav.rules.base import BaseRule
from chav.typing import ColumnType, Diagnostic, Severity, Status

_SUSPICIOUS_NAMES = re.compile(
    r"(status|result|approved|decision|final|outcome|label|flag|verdict|response)",
    re.IGNORECASE,
)


class LabelLeakageRule(BaseRule):
    name = "label_leakage"
    requires_target = True

    def evaluate(
        self,
        profile: DatasetProfile,
        config: ChavConfig,
        compare: CompareProfile | None = None,
        target: str | None = None,
        time_column: str | None = None,
    ) -> Diagnostic:
        if target not in profile.df.columns:
            return self._skip()

        cfg = config.label_leakage
        fail_threshold = cfg["association_threshold"]
        warn_threshold = cfg["warn_threshold"]

        df = profile.df.copy()
        y = df[target]
        feature_cols = [c for c in df.columns if c != target]

        if not feature_cols:
            return self._skip()

        y_clean = y.dropna()
        if len(y_clean) < 10:
            return self._skip()

        assert target is not None
        target_type = profile.column_types.get(target, ColumnType.UNKNOWN)
        is_classification = target_type in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN)
        if not is_classification and target_type == ColumnType.NUMERIC:
            is_classification = y_clean.nunique() <= 20

        valid_idx = y_clean.index
        scores = {}

        for col in feature_cols:
            col_type = profile.column_types.get(col, ColumnType.UNKNOWN)
            if col_type == ColumnType.DATETIME:
                continue

            series = df.loc[valid_idx, col].copy()
            if series.isna().mean() > 0.5:
                continue

            try:
                if col_type == ColumnType.NUMERIC:
                    x = series.fillna(series.median()).values.reshape(-1, 1)
                else:
                    x = series.fillna("__MISSING__").astype("category").cat.codes.values.reshape(-1, 1)

                discrete = col_type != ColumnType.NUMERIC
                if is_classification:
                    mi = mutual_info_classif(x, y_clean, discrete_features=discrete, random_state=42)
                else:
                    mi = mutual_info_regression(x, y_clean, discrete_features=discrete, random_state=42)

                scores[col] = float(mi[0])
            except Exception:
                continue

        if not scores:
            return self._skip()

        max_score = max(scores.values())
        norm_scores = {k: v / max_score for k, v in scores.items()} if max_score > 0 else scores

        suspicious = {}
        for col, norm in sorted(norm_scores.items(), key=lambda x: -x[1]):
            name_boost = 0.05 if _SUSPICIOUS_NAMES.search(col) else 0.0
            adjusted = min(norm + name_boost, 1.0)

            if adjusted >= warn_threshold:
                suspicious[col] = {
                    "mi_raw": round(scores[col], 4),
                    "mi_normalized": round(adjusted, 4),
                    "name_suspicious": bool(_SUSPICIOUS_NAMES.search(col)),
                }

        if not suspicious:
            return Diagnostic(
                rule=self.name,
                status=Status.PASS,
                severity=Severity.LOW,
                confidence=0.9,
                evidence={"top_scores": dict(sorted(norm_scores.items(), key=lambda x: -x[1])[:5])},
            )

        top_col = list(suspicious.keys())[0]
        top_score = suspicious[top_col]["mi_normalized"]

        return Diagnostic(
            rule=self.name,
            status=Status.FAIL if top_score >= fail_threshold else Status.WARN,
            severity=Severity.HIGH if top_score >= fail_threshold else Severity.MEDIUM,
            confidence=min(0.6 + top_score * 0.3, 0.95),
            affected_columns=list(suspicious.keys()),
            evidence=suspicious,
        )
