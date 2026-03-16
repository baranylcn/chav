import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from chav.main import app

client = TestClient(app)


def _make_csv(df: pd.DataFrame) -> io.BytesIO:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


@pytest.fixture
def sample_csv():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "age": np.random.randint(18, 65, n),
        "income": np.random.normal(50000, 15000, n).round(2),
        "city": np.random.choice(["Istanbul", "Ankara", "Izmir"], n),
        "label": np.random.choice([0, 1], n, p=[0.6, 0.4]),
    })
    return _make_csv(df)


@pytest.fixture
def reference_csv():
    np.random.seed(10)
    n = 100
    df = pd.DataFrame({
        "age": np.random.randint(18, 65, n),
        "income": np.random.normal(50000, 15000, n).round(2),
        "city": np.random.choice(["Istanbul", "Ankara", "Izmir"], n),
        "label": np.random.choice([0, 1], n, p=[0.6, 0.4]),
    })
    return _make_csv(df)


class TestHealth:
    def test_health(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestAnalyzeEndpoint:
    def test_single_dataset(self, sample_csv):
        resp = client.post(
            "/analyze",
            files={"data": ("data.csv", sample_csv, "text/csv")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "diagnostics" in body
        assert "dataset_summary" in body
        assert len(body["diagnostics"]) == 10

    def test_with_target(self, sample_csv):
        resp = client.post(
            "/analyze",
            files={"data": ("data.csv", sample_csv, "text/csv")},
            data={"target": "label"},
        )
        assert resp.status_code == 200
        body = resp.json()
        imbalance = next(d for d in body["diagnostics"] if d["rule"] == "imbalance")
        assert imbalance["status"] != "skipped"

    def test_with_reference(self, sample_csv, reference_csv):
        resp = client.post(
            "/analyze",
            files={
                "data": ("current.csv", sample_csv, "text/csv"),
                "reference_data": ("ref.csv", reference_csv, "text/csv"),
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["analysis_context"]["has_reference"] is True
        drift = next(d for d in body["diagnostics"] if d["rule"] == "drift_risk")
        assert drift["status"] != "skipped"

    def test_invalid_file(self):
        resp = client.post(
            "/analyze",
            files={"data": ("data.csv", io.BytesIO(b"not,valid\n\x00\x01"), "text/csv")},
        )
        assert resp.status_code == 200 or resp.status_code == 400
