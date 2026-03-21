from chav.rules.base import BaseRule
from chav.rules.constant_features import ConstantFeaturesRule
from chav.rules.id_like_features import IdLikeFeaturesRule
from chav.rules.duplicate_ingestion import DuplicateIngestionRule
from chav.rules.imbalance import ImbalanceRule
from chav.rules.temporal_inconsistency import TemporalInconsistencyRule
from chav.rules.missing_explosion import MissingExplosionRule
from chav.rules.category_explosion import CategoryExplosionRule
from chav.rules.drift_risk import DriftRiskRule
from chav.rules.feature_instability import FeatureInstabilityRule
from chav.rules.label_leakage import LabelLeakageRule
from chav.rules.structural_missingness import StructuralMissingnessRule
from chav.rules.hidden_redundancy import HiddenRedundancyRule
from chav.rules.conditional_drift import ConditionalDriftRule

ALL_RULES: list[type[BaseRule]] = [
    ConstantFeaturesRule,
    IdLikeFeaturesRule,
    DuplicateIngestionRule,
    ImbalanceRule,
    TemporalInconsistencyRule,
    MissingExplosionRule,
    CategoryExplosionRule,
    DriftRiskRule,
    FeatureInstabilityRule,
    LabelLeakageRule,
    StructuralMissingnessRule,
    HiddenRedundancyRule,
    ConditionalDriftRule,
]
