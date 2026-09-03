"""Alpha signal generation, validation, and model research."""

from qr_haven.alpha.borrow_signal import BorrowRateAlphaSignal
from qr_haven.alpha.combination import (
    combine_scores,
    cross_sectional_zscore,
    information_coefficient,
    pivot_feature_store,
    winsorize,
)
from qr_haven.alpha.decay import (
    ICStability,
    SignalDecayProfile,
    calculate_forward_returns,
    calculate_ic_by_horizon,
    calculate_ic_stability,
    calculate_rolling_ic,
)
from qr_haven.alpha.models import CompositeAlphaModel, RankAlphaModel

__all__ = [
    "BorrowRateAlphaSignal",
    "CompositeAlphaModel",
    "ICStability",
    "RankAlphaModel",
    "SignalDecayProfile",
    "calculate_forward_returns",
    "calculate_ic_by_horizon",
    "calculate_ic_stability",
    "calculate_rolling_ic",
    "combine_scores",
    "cross_sectional_zscore",
    "information_coefficient",
    "pivot_feature_store",
    "winsorize",
]
