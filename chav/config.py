from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChavConfig:
    label_leakage: dict = field(default_factory=lambda: {
        "association_threshold": 0.9,
        "warn_threshold": 0.7,
        "suspicious_name_patterns": [
            "status", "result", "approved", "decision", "final", "outcome",
        ],
    })
    duplicate_ingestion: dict = field(default_factory=lambda: {
        "duplicate_ratio_threshold": 0.05,
        "warn_ratio_threshold": 0.01,
    })
    category_explosion: dict = field(default_factory=lambda: {
        "cardinality_growth_threshold": 2.0,
        "unseen_ratio_threshold": 0.5,
        "warn_growth_threshold": 1.5,
    })
    drift_risk: dict = field(default_factory=lambda: {
        "psi_fail_threshold": 0.25,
        "psi_warn_threshold": 0.1,
        "ks_pvalue_threshold": 0.01,
    })
    feature_instability: dict = field(default_factory=lambda: {
        "instability_fail_threshold": 0.6,
        "instability_warn_threshold": 0.3,
    })
    missing_explosion: dict = field(default_factory=lambda: {
        "absolute_delta_threshold": 0.1,
        "relative_multiplier_threshold": 3.0,
    })
    constant_features: dict = field(default_factory=lambda: {
        "near_constant_threshold": 0.99,
    })
    id_like_features: dict = field(default_factory=lambda: {
        "uniqueness_threshold": 0.8,
        "min_rows": 20,
    })
    temporal_inconsistency: dict = field(default_factory=lambda: {
        "future_tolerance_hours": 24,
        "ancient_year_floor": 1970,
        "parse_failure_threshold": 0.05,
    })
    imbalance: dict = field(default_factory=lambda: {
        "imbalance_ratio_threshold": 10.0,
        "warn_ratio_threshold": 5.0,
        "minority_share_threshold": 0.05,
    })
