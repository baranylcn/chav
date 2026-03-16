from __future__ import annotations

from chav.rules.base import BaseRule
from chav.typing import Diagnostic, Status, Severity, ColumnType
from chav.config import ChavConfig
from chav.profiling.dataset_profile import DatasetProfile
from chav.profiling.compare_profile import CompareProfile

MAX_CLASSES = 50


class ImbalanceRule(BaseRule):
    name = "imbalance"
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

        target_type = profile.column_types.get(target, ColumnType.UNKNOWN)
        if target_type == ColumnType.NUMERIC:
            return self._skip()

        cfg = config.imbalance
        fail_ratio = cfg["imbalance_ratio_threshold"]
        warn_ratio = cfg["warn_ratio_threshold"]
        min_share = cfg["minority_share_threshold"]

        series = profile.df[target].dropna()
        vc = series.value_counts()

        if len(vc) < 2 or len(vc) > MAX_CLASSES:
            return self._skip()

        majority_count = int(vc.iloc[0])
        minority_count = int(vc.iloc[-1])
        imbalance_ratio = majority_count / minority_count if minority_count > 0 else float('inf')
        minority_share = minority_count / len(series) if len(series) > 0 else 0.0

        evidence = {
            "n_classes": len(vc),
            "majority_class": str(vc.index[0]),
            "majority_count": majority_count,
            "minority_class": str(vc.index[-1]),
            "minority_count": minority_count,
            "imbalance_ratio": round(imbalance_ratio, 2),
            "minority_share": round(minority_share, 4),
        }

        if imbalance_ratio >= fail_ratio or minority_share < min_share:
            return Diagnostic(
                rule=self.name,
                status=Status.FAIL,
                severity=Severity.HIGH,
                confidence=min(0.7 + (1 - minority_share), 1.0),
                affected_columns=[target],
                evidence=evidence,
            )
        elif imbalance_ratio >= warn_ratio:
            return Diagnostic(
                rule=self.name,
                status=Status.WARN,
                severity=Severity.MEDIUM,
                confidence=0.7,
                affected_columns=[target],
                evidence=evidence,
            )
        else:
            return Diagnostic(
                rule=self.name,
                status=Status.PASS,
                severity=Severity.LOW,
                confidence=1.0,
                affected_columns=[target],
                evidence=evidence,
            )
