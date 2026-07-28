"""Reusable feature engineering and feature-store interfaces."""

from qr_haven.features.reference import (
    lagged_return,
    reference_feature_registry,
    rolling_momentum,
    rolling_volatility,
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
    "compute_features",
    "lagged_return",
    "reference_feature_registry",
    "rolling_momentum",
    "rolling_volatility",
]
