"""Dynamic Borrow Demand Surface — public API.

Modules
-------
features          : RawFeatures, SurfaceFeatures, FeaturePipeline
model             : BorrowDemandConfig, BorrowDemandSurface
trainer           : TrainingResult, train_surface, make_model_and_likelihood
updater           : LocateEvent, OnlineSurfaceUpdater
calibration       : CalibrationResult, DemandRateCalibrator
allocator         : LocateRequest, AllocationResult, InventorySnapshot, LocateAllocator
diagnostics       : SurfaceMetrics, CalibrationDiagnostic, ShortageRecallMetrics,
                    surface_rmse, calibration_reliability, shortage_recall
inference         : shortage_probability, demand_quantile
si_proxy          : SIAnchor, SIProxyResult, RealTimeSIProxy
earnings_forecast : EarningsRecord, EarningsForecastResult, EarningsPanelBuilder,
                    PanelRegressionForecaster, LSTMFeeForecaster, LSTMFitResult,
                    EarningsWindowForecaster
regime_transition : BorrowRegime, RegimeTransitionResult, EmpiricalTransitionMatrix,
                    LogisticTransitionForecaster, BorrowRegimeClassifier
"""

from qr_haven.borrow_demand.features import (
    N_FEATURES,
    N_FULL_INPUT_DIMS,
    FEATURE_NAMES,
    RawFeatures,
    SurfaceFeatures,
    FeaturePipeline,
)
from qr_haven.borrow_demand.model import BorrowDemandConfig, BorrowDemandSurface
from qr_haven.borrow_demand.trainer import (
    TrainingResult,
    train_surface,
    make_model_and_likelihood,
)
from qr_haven.borrow_demand.updater import LocateEvent, OnlineSurfaceUpdater
from qr_haven.borrow_demand.calibration import CalibrationResult, DemandRateCalibrator
from qr_haven.borrow_demand.allocator import (
    LocateRequest,
    AllocationResult,
    InventorySnapshot,
    LocateAllocator,
)
from qr_haven.borrow_demand.diagnostics import (
    SurfaceMetrics,
    CalibrationDiagnostic,
    ShortageRecallMetrics,
    surface_rmse,
    calibration_reliability,
    shortage_recall,
)
from qr_haven.borrow_demand.inference import shortage_probability, demand_quantile
from qr_haven.borrow_demand.si_proxy import SIAnchor, SIProxyResult, RealTimeSIProxy
from qr_haven.borrow_demand.earnings_forecast import (
    EarningsRecord,
    EarningsForecastResult,
    EarningsPanelBuilder,
    PanelRegressionForecaster,
    LSTMFeeForecaster,
    LSTMFitResult,
    EarningsWindowForecaster,
)
from qr_haven.borrow_demand.regime_transition import (
    BorrowRegime,
    RegimeTransitionResult,
    EmpiricalTransitionMatrix,
    LogisticTransitionForecaster,
    BorrowRegimeClassifier,
)

__all__ = [
    # features
    "N_FEATURES",
    "N_FULL_INPUT_DIMS",
    "FEATURE_NAMES",
    "RawFeatures",
    "SurfaceFeatures",
    "FeaturePipeline",
    # model
    "BorrowDemandConfig",
    "BorrowDemandSurface",
    # trainer
    "TrainingResult",
    "train_surface",
    "make_model_and_likelihood",
    # updater
    "LocateEvent",
    "OnlineSurfaceUpdater",
    # calibration
    "CalibrationResult",
    "DemandRateCalibrator",
    # allocator
    "LocateRequest",
    "AllocationResult",
    "InventorySnapshot",
    "LocateAllocator",
    # diagnostics
    "SurfaceMetrics",
    "CalibrationDiagnostic",
    "ShortageRecallMetrics",
    "surface_rmse",
    "calibration_reliability",
    "shortage_recall",
    # inference
    "shortage_probability",
    "demand_quantile",
    # si_proxy
    "SIAnchor",
    "SIProxyResult",
    "RealTimeSIProxy",
    # earnings_forecast
    "EarningsRecord",
    "EarningsForecastResult",
    "EarningsPanelBuilder",
    "PanelRegressionForecaster",
    "LSTMFeeForecaster",
    "LSTMFitResult",
    "EarningsWindowForecaster",
    # regime_transition
    "BorrowRegime",
    "RegimeTransitionResult",
    "EmpiricalTransitionMatrix",
    "LogisticTransitionForecaster",
    "BorrowRegimeClassifier",
]
