import numpy as np
import pandas as pd
import pytest

from chav import analyze
from chav.typing import Status


class TestConstantFeatures:
    def test_detects_constant(self, dirty_df):
        report = analyze(dirty_df)
        diag = next(d for d in report.diagnostics if d.rule == "constant_features")
        assert diag.status in (Status.FAIL, Status.WARN)
        assert "constant_col" in diag.affected_columns

    def test_passes_clean(self, clean_df):
        report = analyze(clean_df)
        diag = next(d for d in report.diagnostics if d.rule == "constant_features")
        assert diag.status == Status.PASS


class TestIdLikeFeatures:
    def test_detects_id_column(self, dirty_df):
        report = analyze(dirty_df)
        diag = next(d for d in report.diagnostics if d.rule == "id_like_features")
        assert diag.status == Status.WARN
        assert "user_id" in diag.affected_columns

    def test_passes_clean(self, clean_df):
        report = analyze(clean_df)
        diag = next(d for d in report.diagnostics if d.rule == "id_like_features")
        assert diag.status == Status.PASS


class TestDuplicateIngestion:
    def test_detects_duplicates(self, dirty_df):
        report = analyze(dirty_df)
        diag = next(d for d in report.diagnostics if d.rule == "duplicate_ingestion")
        assert diag.status in (Status.FAIL, Status.WARN)

    def test_passes_clean(self, clean_df):
        report = analyze(clean_df)
        diag = next(d for d in report.diagnostics if d.rule == "duplicate_ingestion")
        assert diag.status == Status.PASS


class TestImbalance:
    def test_detects_imbalance(self, dirty_df):
        report = analyze(dirty_df, target="label")
        diag = next(d for d in report.diagnostics if d.rule == "imbalance")
        assert diag.status in (Status.FAIL, Status.WARN)

    def test_passes_balanced(self, clean_df):
        report = analyze(clean_df, target="label")
        diag = next(d for d in report.diagnostics if d.rule == "imbalance")
        assert diag.status == Status.PASS

    def test_skips_without_target(self, clean_df):
        report = analyze(clean_df)
        diag = next(d for d in report.diagnostics if d.rule == "imbalance")
        assert diag.status == Status.SKIPPED


class TestTemporalInconsistency:
    def test_detects_future_timestamps(self):
        df = pd.DataFrame({
            "event_time": pd.date_range("2099-01-01", periods=100, freq="h"),
            "value": range(100),
        })
        report = analyze(df, time_column="event_time")
        diag = next(d for d in report.diagnostics if d.rule == "temporal_inconsistency")
        assert diag.status in (Status.FAIL, Status.WARN)

    def test_passes_clean(self, clean_df):
        report = analyze(clean_df, time_column="signup_date")
        diag = next(d for d in report.diagnostics if d.rule == "temporal_inconsistency")
        assert diag.status == Status.PASS

    def test_skips_without_time_column(self, clean_df):
        report = analyze(clean_df)
        diag = next(d for d in report.diagnostics if d.rule == "temporal_inconsistency")
        assert diag.status == Status.SKIPPED


class TestMissingExplosion:
    def test_detects_missing_spike(self, reference_df, drifted_missing_df):
        report = analyze(drifted_missing_df, reference_data=reference_df)
        diag = next(d for d in report.diagnostics if d.rule == "missing_explosion")
        assert diag.status == Status.FAIL
        assert "income" in diag.affected_columns

    def test_skips_without_reference(self, clean_df):
        report = analyze(clean_df)
        diag = next(d for d in report.diagnostics if d.rule == "missing_explosion")
        assert diag.status == Status.SKIPPED


class TestCategoryExplosion:
    def test_detects_explosion(self, reference_df, drifted_df):
        report = analyze(drifted_df, reference_data=reference_df)
        diag = next(d for d in report.diagnostics if d.rule == "category_explosion")
        assert diag.status in (Status.FAIL, Status.WARN)
        assert "city" in diag.affected_columns

    def test_skips_without_reference(self, clean_df):
        report = analyze(clean_df)
        diag = next(d for d in report.diagnostics if d.rule == "category_explosion")
        assert diag.status == Status.SKIPPED


class TestDriftRisk:
    def test_detects_drift(self, reference_df, drifted_df):
        report = analyze(drifted_df, reference_data=reference_df)
        diag = next(d for d in report.diagnostics if d.rule == "drift_risk")
        assert diag.status in (Status.FAIL, Status.WARN)

    def test_skips_without_reference(self, clean_df):
        report = analyze(clean_df)
        diag = next(d for d in report.diagnostics if d.rule == "drift_risk")
        assert diag.status == Status.SKIPPED


class TestFeatureInstability:
    def test_detects_instability(self, reference_df, drifted_df):
        report = analyze(drifted_df, reference_data=reference_df)
        diag = next(d for d in report.diagnostics if d.rule == "feature_instability")
        assert diag.status in (Status.FAIL, Status.WARN)

    def test_skips_without_reference(self, clean_df):
        report = analyze(clean_df)
        diag = next(d for d in report.diagnostics if d.rule == "feature_instability")
        assert diag.status == Status.SKIPPED


class TestLabelLeakage:
    def test_detects_leakage(self, dirty_df):
        report = analyze(dirty_df, target="label")
        diag = next(d for d in report.diagnostics if d.rule == "label_leakage")
        assert diag.status in (Status.FAIL, Status.WARN)
        assert "approval_status" in diag.affected_columns

    def test_skips_without_target(self, clean_df):
        report = analyze(clean_df)
        diag = next(d for d in report.diagnostics if d.rule == "label_leakage")
        assert diag.status == Status.SKIPPED


class TestReportStructure:
    def test_to_dict_structure(self, clean_df):
        report = analyze(clean_df, target="label", time_column="signup_date")
        d = report.to_dict(all=True)
        assert "dataset_summary" in d
        assert "analysis_context" in d
        assert "diagnostics" in d
        assert "counts" in d
        assert d["dataset_summary"]["rows"] == len(clean_df)
        assert d["dataset_summary"]["columns"] == len(clean_df.columns)
        assert len(d["diagnostics"]) == 10

    def test_to_json(self, clean_df):
        report = analyze(clean_df)
        j = report.to_json()
        import json
        parsed = json.loads(j)
        assert "diagnostics" in parsed

    def test_summary_string(self, clean_df):
        report = analyze(clean_df)
        s = report.summary()
        assert "Chav Report" in s
