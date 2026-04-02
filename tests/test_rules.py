import numpy as np
import pandas as pd

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
        df = pd.DataFrame(
            {
                "event_time": pd.date_range("2099-01-01", periods=100, freq="h"),
                "value": range(100),
            }
        )
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


class TestStructuralMissingness:
    def test_detects_correlated_nulls(self):
        np.random.seed(42)
        n = 300
        df = pd.DataFrame(
            {
                "a": np.random.normal(0, 1, n),
                "b": np.random.normal(0, 1, n),
                "c": np.random.normal(0, 1, n),
            }
        )
        mask = np.random.choice([True, False], n, p=[0.3, 0.7])
        df.loc[mask, "a"] = np.nan
        df.loc[mask, "b"] = np.nan

        report = analyze(df)
        diag = next(d for d in report.diagnostics if d.rule == "structural_missingness")
        assert diag.status in (Status.FAIL, Status.WARN)
        assert "a" in diag.affected_columns
        assert "b" in diag.affected_columns

    def test_passes_independent_nulls(self):
        np.random.seed(42)
        n = 300
        df = pd.DataFrame(
            {
                "a": np.random.normal(0, 1, n),
                "b": np.random.normal(0, 1, n),
            }
        )
        df.loc[np.random.choice(n, 30, replace=False), "a"] = np.nan
        df.loc[np.random.choice(n, 30, replace=False), "b"] = np.nan

        report = analyze(df)
        diag = next(d for d in report.diagnostics if d.rule == "structural_missingness")
        assert diag.status == Status.PASS

    def test_passes_no_nulls(self, clean_df):
        report = analyze(clean_df)
        diag = next(d for d in report.diagnostics if d.rule == "structural_missingness")
        assert diag.status == Status.PASS


class TestHiddenRedundancy:
    def test_detects_numeric_redundancy(self):
        np.random.seed(42)
        n = 200
        base = np.random.normal(0, 1, n)
        df = pd.DataFrame(
            {
                "x": base,
                "y": base * 2 + 1,
                "z": np.random.normal(0, 1, n),
            }
        )
        report = analyze(df)
        diag = next(d for d in report.diagnostics if d.rule == "hidden_redundancy")
        assert diag.status in (Status.FAIL, Status.WARN)
        assert "x" in diag.affected_columns
        assert "y" in diag.affected_columns

    def test_detects_categorical_redundancy(self):
        np.random.seed(42)
        n = 200
        codes = np.random.choice(["A", "B", "C"], n)
        df = pd.DataFrame(
            {
                "col1": codes,
                "col2": [{"A": "X", "B": "Y", "C": "Z"}[c] for c in codes],
                "value": np.random.normal(0, 1, n),
            }
        )
        report = analyze(df)
        diag = next(d for d in report.diagnostics if d.rule == "hidden_redundancy")
        assert diag.status in (Status.FAIL, Status.WARN)
        assert "col1" in diag.affected_columns
        assert "col2" in diag.affected_columns

    def test_passes_independent_columns(self, clean_df):
        report = analyze(clean_df)
        diag = next(d for d in report.diagnostics if d.rule == "hidden_redundancy")
        assert diag.status == Status.PASS


class TestConditionalDrift:
    def test_detects_hidden_subgroup_drift(self):
        np.random.seed(42)
        n = 1000
        ref_df = pd.DataFrame(
            {
                "segment": np.random.choice(["A", "B"], n, p=[0.85, 0.15]),
                "value": np.random.normal(50, 10, n),
            }
        )
        cur_segments = np.random.choice(["A", "B"], n, p=[0.85, 0.15])
        cur_df = pd.DataFrame(
            {
                "segment": cur_segments,
                "value": np.where(
                    cur_segments == "A",
                    np.random.normal(50, 10, n),
                    np.random.normal(75, 10, n),
                ),
            }
        )

        report = analyze(cur_df, reference_data=ref_df)
        diag = next(d for d in report.diagnostics if d.rule == "conditional_drift")
        assert diag.status in (Status.FAIL, Status.WARN)
        assert "segment" in diag.affected_columns
        assert "value" in diag.affected_columns

    def test_passes_no_drift(self, reference_df):
        report = analyze(reference_df, reference_data=reference_df)
        diag = next(d for d in report.diagnostics if d.rule == "conditional_drift")
        assert diag.status == Status.PASS

    def test_detects_disproportionate_drift(self):
        np.random.seed(42)
        n = 1000
        ref_segments = np.random.choice(["A", "B"], n, p=[0.7, 0.3])
        ref_df = pd.DataFrame(
            {
                "segment": ref_segments,
                "value": np.where(
                    ref_segments == "A",
                    np.random.normal(50, 10, n),
                    np.random.normal(55, 10, n),
                ),
            }
        )
        cur_segments = np.random.choice(["A", "B"], n, p=[0.7, 0.3])
        cur_df = pd.DataFrame(
            {
                "segment": cur_segments,
                "value": np.where(
                    cur_segments == "A",
                    np.random.normal(52, 10, n),
                    np.random.normal(80, 10, n),
                ),
            }
        )

        report = analyze(cur_df, reference_data=ref_df)
        diag = next(d for d in report.diagnostics if d.rule == "conditional_drift")
        assert diag.status in (Status.FAIL, Status.WARN)
        has_disprop = any(
            s.get("drift_type") == "disproportionate"
            for f in diag.evidence.get("findings", [])
            for s in f.get("subgroups", [])
        )
        assert has_disprop

    def test_works_with_small_dataset(self):
        np.random.seed(42)
        n = 80
        ref_df = pd.DataFrame(
            {
                "group": np.random.choice(["X", "Y"], n, p=[0.6, 0.4]),
                "metric": np.random.normal(100, 10, n),
            }
        )
        cur_groups = np.random.choice(["X", "Y"], n, p=[0.6, 0.4])
        cur_df = pd.DataFrame(
            {
                "group": cur_groups,
                "metric": np.where(
                    cur_groups == "X",
                    np.random.normal(100, 10, n),
                    np.random.normal(130, 10, n),
                ),
            }
        )

        report = analyze(cur_df, reference_data=ref_df)
        diag = next(d for d in report.diagnostics if d.rule == "conditional_drift")
        assert diag.status in (Status.FAIL, Status.WARN, Status.PASS)
        assert diag.status != Status.ERROR

    def test_skips_without_reference(self, clean_df):
        report = analyze(clean_df)
        diag = next(d for d in report.diagnostics if d.rule == "conditional_drift")
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
        assert len(d["diagnostics"]) == 13

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

    def test_to_dataframe(self, clean_df):
        report = analyze(clean_df)
        df = report.to_dataframe(all=True)
        assert list(df.columns) == ["rule", "status", "severity", "confidence", "affected_columns", "evidence"]
        assert len(df) == len(report.diagnostics)

    def test_to_dataframe_filters_actionable(self, dirty_df):
        report = analyze(dirty_df)
        df_all = report.to_dataframe(all=True)
        df_actionable = report.to_dataframe(all=False)
        assert len(df_actionable) <= len(df_all)
        assert set(df_actionable["status"]).issubset({"warn", "fail", "error"})

    def test_to_csv_returns_string(self, clean_df):
        import io

        report = analyze(clean_df)
        csv_str = report.to_csv(all=True)
        assert csv_str is not None
        df = pd.read_csv(io.StringIO(csv_str))
        assert "rule" in df.columns

    def test_to_csv_writes_file(self, clean_df, tmp_path):
        report = analyze(clean_df)
        path = str(tmp_path / "report.csv")
        result = report.to_csv(path=path, all=True)
        assert result is None
        df = pd.read_csv(path)
        assert "rule" in df.columns

    def test_to_excel_writes_file(self, clean_df, tmp_path):
        report = analyze(clean_df)
        path = str(tmp_path / "report.xlsx")
        report.to_excel(path, all=True)
        df = pd.read_excel(path)
        assert "rule" in df.columns
