"""Research methodology, experiment definitions, and reproducible workflows."""

from qr_haven.research.capacity import CapacityCurve, CapacityPoint, sweep_capacity
from qr_haven.research.pipeline import (
    CostBreakdown,
    PipelineConfig,
    PipelineResult,
    ResearchPipeline,
)

__all__ = [
    "CapacityCurve",
    "CapacityPoint",
    "CostBreakdown",
    "PipelineConfig",
    "PipelineResult",
    "ResearchPipeline",
    "sweep_capacity",
]
