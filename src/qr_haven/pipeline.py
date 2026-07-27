"""Composable research workflow orchestration."""

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from qr_haven.interfaces import AlphaModel, PortfolioOptimizer, RiskEngine


class ResearchStep(Protocol):
    """A named transformation in a research workflow."""

    name: str

    def run(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Transform a research dataframe."""


@dataclass
class ResearchPipeline:
    """Minimal pipeline shell for repeatable hypothesis-to-results workflows."""

    steps: list[ResearchStep]

    def run(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Run each step in sequence."""

        result = frame.copy()
        for step in self.steps:
            result = step.run(result)
        return result


@dataclass(frozen=True)
class ModelPortfolioWorkflow:
    """Connect alpha, optimization, and risk evaluation for a single research pass."""

    alpha_model: AlphaModel
    optimizer: PortfolioOptimizer
    risk_engine: RiskEngine

