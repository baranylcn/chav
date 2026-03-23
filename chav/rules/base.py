from __future__ import annotations

from abc import ABC, abstractmethod

from chav.config import ChavConfig
from chav.profiling.compare_profile import CompareProfile
from chav.profiling.dataset_profile import DatasetProfile
from chav.typing import Diagnostic, Severity, Status


class BaseRule(ABC):
    name: str = ""
    requires_reference: bool = False
    requires_target: bool = False
    requires_time_column: bool = False

    @abstractmethod
    def evaluate(
        self,
        profile: DatasetProfile,
        config: ChavConfig,
        compare: CompareProfile | None = None,
        target: str | None = None,
        time_column: str | None = None,
    ) -> Diagnostic: ...

    def _skip(self) -> Diagnostic:
        return Diagnostic(
            rule=self.name,
            status=Status.SKIPPED,
            severity=Severity.LOW,
            confidence=0.0,
        )

    def _error(self, error: str) -> Diagnostic:
        return Diagnostic(
            rule=self.name,
            status=Status.ERROR,
            severity=Severity.LOW,
            confidence=0.0,
            evidence={"error": error},
        )

    def safe_evaluate(
        self,
        profile: DatasetProfile,
        config: ChavConfig,
        compare: CompareProfile | None = None,
        target: str | None = None,
        time_column: str | None = None,
    ) -> Diagnostic:
        try:
            if self.requires_reference and compare is None:
                return self._skip()
            if self.requires_target and target is None:
                return self._skip()
            if self.requires_time_column and time_column is None:
                return self._skip()
            return self.evaluate(profile, config, compare, target, time_column)
        except Exception as e:
            return self._error(str(e))
