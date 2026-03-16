import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def clean_df():
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "age": np.random.randint(18, 65, n),
        "income": np.random.normal(50000, 15000, n).round(2),
        "city": np.random.choice(["Istanbul", "Ankara", "Izmir"], n),
        "signup_date": pd.date_range("2024-01-01", periods=n, freq="h"),
        "is_active": np.random.choice([True, False], n),
        "label": np.random.choice([0, 1], n, p=[0.6, 0.4]),
    })


@pytest.fixture
def dirty_df():
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "user_id": [f"USR-{i:06d}" for i in range(n)],
        "age": np.random.randint(18, 65, n),
        "income": np.random.normal(50000, 15000, n).round(2),
        "city": np.random.choice(["Istanbul", "Ankara", "Izmir"], n),
        "constant_col": ["same_value"] * n,
        "approval_status": None,
        "signup_date": pd.date_range("2024-01-01", periods=n, freq="h"),
        "label": None,
    })
    df["label"] = np.random.choice([0, 1], n, p=[0.9, 0.1])
    df["approval_status"] = df["label"].map({0: "rejected", 1: "approved"})
    dup_rows = df.head(30).copy()
    df = pd.concat([df, dup_rows], ignore_index=True)
    return df


@pytest.fixture
def reference_df():
    np.random.seed(10)
    n = 200
    return pd.DataFrame({
        "age": np.random.randint(18, 65, n),
        "income": np.random.normal(50000, 15000, n).round(2),
        "city": np.random.choice(["Istanbul", "Ankara", "Izmir"], n),
        "score": np.random.uniform(0, 1, n),
    })


@pytest.fixture
def drifted_df():
    np.random.seed(99)
    n = 200
    return pd.DataFrame({
        "age": np.random.randint(40, 90, n),
        "income": np.random.normal(80000, 30000, n).round(2),
        "city": np.random.choice(["Istanbul", "Ankara", "Izmir", "Bursa", "Antalya",
                                   "Trabzon", "Konya", "Adana", "Mersin", "Kayseri"], n),
        "score": np.random.uniform(0, 1, n),
    })


@pytest.fixture
def drifted_missing_df():
    np.random.seed(99)
    n = 200
    df = pd.DataFrame({
        "age": np.random.randint(18, 65, n),
        "income": np.random.normal(50000, 15000, n).round(2),
        "city": np.random.choice(["Istanbul", "Ankara", "Izmir"], n),
        "score": np.random.uniform(0, 1, n),
    })
    df.loc[df.index[:80], "income"] = np.nan
    df.loc[df.index[:60], "city"] = np.nan
    return df
