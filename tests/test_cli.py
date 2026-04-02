from __future__ import annotations

import csv
import io
import json

import pandas as pd
import pytest

from chav.cli import main


@pytest.fixture
def csv_file(tmp_path):
    df = pd.DataFrame(
        {
            "age": [25, 30, 35, 40],
            "income": [50000, 60000, 70000, 80000],
            "city": ["Istanbul", "Ankara", "Izmir", "Bursa"],
            "label": [0, 1, 0, 1],
        }
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def reference_csv_file(tmp_path):
    df = pd.DataFrame(
        {
            "age": [20, 25, 30],
            "income": [40000, 50000, 60000],
            "city": ["Istanbul", "Ankara", "Izmir"],
            "label": [0, 1, 0],
        }
    )
    path = tmp_path / "reference.csv"
    df.to_csv(path, index=False)
    return str(path)


class TestCLIBasic:
    def test_no_command_returns_1(self):
        assert main([]) == 1

    def test_analyze_returns_0(self, csv_file):
        assert main(["analyze", csv_file]) == 0

    def test_analyze_with_target(self, csv_file):
        assert main(["analyze", csv_file, "--target", "label"]) == 0

    def test_analyze_with_reference(self, csv_file, reference_csv_file):
        assert main(["analyze", csv_file, "--reference", reference_csv_file]) == 0

    def test_analyze_with_all_options(self, csv_file, reference_csv_file):
        assert main(["analyze", csv_file, "--reference", reference_csv_file, "--target", "label", "--all"]) == 0

    def test_invalid_file_returns_1(self):
        assert main(["analyze", "nonexistent.csv"]) == 1


class TestCLIFormats:
    def test_format_json(self, csv_file, capsys):
        assert main(["analyze", csv_file, "--format", "json", "--all"]) == 0
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert "diagnostics" in parsed

    def test_format_csv(self, csv_file, capsys):
        assert main(["analyze", csv_file, "--format", "csv", "--all"]) == 0
        out = capsys.readouterr().out
        reader = csv.DictReader(io.StringIO(out))
        rows = list(reader)
        assert len(rows) > 0
        assert "rule" in rows[0]

    def test_format_summary(self, csv_file, capsys):
        assert main(["analyze", csv_file, "--format", "summary"]) == 0
        out = capsys.readouterr().out
        assert "Chav Report" in out


class TestCLIOutputFile:
    def test_output_file_written(self, csv_file, tmp_path):
        out_path = str(tmp_path / "out.json")
        assert main(["analyze", csv_file, "--format", "json", "--output-file", out_path]) == 0
        with open(out_path, encoding="utf-8") as f:
            parsed = json.loads(f.read())
        assert "diagnostics" in parsed
