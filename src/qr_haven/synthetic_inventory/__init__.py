"""Synthetic inventory creation optimizer.

Build economically equivalent positions when physical inventory is scarce.
A network-flow optimizer traverses a transformation graph of equities, ETFs,
futures, options, swaps, and repo to find the cheapest replication path subject
to exact exposure equivalence.
"""

from qr_haven.synthetic_inventory.costs import CostBreakdown, CostConfig
from qr_haven.synthetic_inventory.graph import TransformationGraph
from qr_haven.synthetic_inventory.instruments import InstrumentSpec, InstrumentType
from qr_haven.synthetic_inventory.optimizer import (
    OptimizationMode,
    OptimizationResult,
    SyntheticInventoryOptimizer,
    SyntheticPathResult,
)

__all__ = [
    "CostBreakdown",
    "CostConfig",
    "InstrumentSpec",
    "InstrumentType",
    "OptimizationMode",
    "OptimizationResult",
    "SyntheticInventoryOptimizer",
    "SyntheticPathResult",
    "TransformationGraph",
]
