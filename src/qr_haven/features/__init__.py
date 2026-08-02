"""Reusable feature engineering and feature-store interfaces."""

from qr_haven.features.reference import (
    lagged_return,
    reference_feature_registry,
    rolling_momentum,
    rolling_volatility,
)
from qr_haven.features.securities_lending import (
    borrow_fee_rate,
    days_to_cover,
    securities_lending_feature_registry,
    short_interest_change,
    short_interest_ratio,
    utilization_rate,
)
from qr_haven.features.store import (
    FEATURE_OUTPUT_COLUMNS,
    FeatureDefinition,
    FeatureRegistry,
    FeatureTransform,
    compute_features,
)

__all__ = [
    "FEATURE_OUTPUT_COLUMNS",
    "FeatureDefinition",
    "FeatureRegistry",
    "FeatureTransform",
    "borrow_fee_rate",
    "compute_features",
    "days_to_cover",
    "lagged_return",
    "reference_feature_registry",
    "rolling_momentum",
    "rolling_volatility",
    "securities_lending_feature_registry",
    "short_interest_change",
    "short_interest_ratio",
    "utilization_rate",
]
