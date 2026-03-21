# Chav

Lightweight diagnostic engine for tabular datasets. One function call, structured results.

Chav inspects your data and returns actionable diagnostics about hidden risks, not just raw metrics. It tells you which columns are involved, how severe the issue is, and how confident the detection is.

## Install

```bash
git clone https://github.com/baranylcn/chav.git
cd chav
pip install -e .
```

## Quick Start

```python
from chav import analyze

report = analyze("data.csv", target="label")

print(report.summary())
```

```
>>> print(report.summary())

Chav Report | 506 rows x 14 cols
1 fail, 1 warn, 4 pass, 7 skipped

  FAIL  label_leakage         high    0.90   [LSTAT, RM, INDUS]
  WARN  hidden_redundancy     medium  0.91   [RAD, TAX]
```

## How It Works

Chav runs a fixed set of diagnostic rules against your dataset. Each rule checks for a specific class of data quality issue and returns a structured result.

```python
report = analyze(
    data=df,                      # DataFrame or CSV path
    reference_data=previous_df,   # optional: baseline dataset for comparison
    target="label_column",        # optional: target column for leakage/imbalance checks
    time_column="event_time",     # optional: time column for temporal checks
)
```

### Two Modes

**Single-dataset mode** - diagnostics computed from one dataset alone:

- Label leakage, duplicate ingestion, constant features, id-like features, temporal inconsistency, imbalance, structural missingness, hidden redundancy

**Compare mode** - requires a reference dataset:

- Category explosion, drift risk, feature instability, missing explosion, conditional drift

Rules that require unavailable inputs are automatically skipped.

## Diagnostic Rules

| Rule | What it detects |
|---|---|
| `label_leakage` | Features suspiciously predictive of the target |
| `duplicate_ingestion` | Rows duplicated from replay, retry, or pipeline errors |
| `constant_features` | Columns with no meaningful variation |
| `id_like_features` | Columns that look like identifiers, not analytical variables |
| `temporal_inconsistency` | Future timestamps, parse failures, suspicious time patterns |
| `imbalance` | Severe class imbalance in the target column |
| `structural_missingness` | Correlated null patterns between column pairs |
| `hidden_redundancy` | Column pairs carrying nearly identical information |
| `category_explosion` | Abnormal growth in distinct category count vs. reference |
| `drift_risk` | Distribution shift between reference and current data |
| `feature_instability` | Features whose statistical behavior changed enough to be unreliable |
| `missing_explosion` | Sudden increase in missing values vs. reference |
| `conditional_drift` | Distribution shift hidden within subgroups while overall looks stable |

## Response Structure

### Report

```python
report.summary()   # human-readable text
report.to_dict()   # structured dict (actionable results only)
report.to_dict(all=True)  # all results including pass/skipped
report.to_json()   # JSON string
```

`to_dict()` returns:

```json
{
  "dataset_summary": { "rows": 506, "columns": 14 },
  "analysis_context": { "has_reference": false, "target": "label", "time_column": null },
  "diagnostics": [ ... ],
  "counts": { "pass": 4, "warn": 1, "fail": 1, "skipped": 7, "error": 0 }
}
```

### Diagnostic Object

Each diagnostic contains:

| Field | Type | Meaning |
|---|---|---|
| `rule` | string | Rule identifier |
| `status` | string | `pass`, `warn`, `fail`, `skipped`, or `error` |
| `severity` | string | `low`, `medium`, or `high` |
| `confidence` | float | 0–1, strength of evidence supporting the finding |
| `affected_columns` | list | Columns involved in the issue |
| `evidence` | dict | Rule-specific metrics and details |

**Status meanings:**

- `pass` - no issue detected
- `warn` - issue exists but is moderate or uncertain
- `fail` - issue requires attention
- `skipped` - missing required input (e.g., no target column provided)
- `error` - rule encountered an internal error

**Severity** reflects likely impact, not just metric size. Label leakage suspicion is always `high`. A mild drift in a single column may be `medium` or `low`.

**Confidence** is not a probability. It is a deterministic measure of how strongly the observed evidence supports the rule's finding.

## Configuration

All thresholds are configurable. Defaults are sensible for most datasets.

```python
from chav import analyze, ChavConfig

config = ChavConfig()
config.drift_risk["psi_fail_threshold"] = 0.3
config.hidden_redundancy["correlation_fail_threshold"] = 0.98

report = analyze(df, reference_data=ref_df, config=config)
```

## Column Type Overrides

Chav infers column types automatically. You can override when needed:

```python
from chav import analyze, ColumnType

report = analyze(df, type_overrides={"zip_code": ColumnType.CATEGORICAL})
```

## Development

```bash
git clone https://github.com/baranylcn/chav.git
cd chav
pip install -e ".[dev]"
pytest
```

### Adding a New Rule

1. Create `chav/rules/your_rule.py` - subclass `BaseRule`, implement `evaluate()`
2. Add default thresholds to `ChavConfig` in `config.py`
3. Register the class in `rules/__init__.py` -> `ALL_RULES`
4. Add tests in `tests/test_rules.py`

Every rule must return a `Diagnostic` with status, severity, confidence, affected columns, and evidence. If a rule fails internally, it should not crash the analysis - `safe_evaluate()` handles this automatically.

## Contributing

Contributions are welcome. Please open an issue before submitting large changes.
