"""Strict, versioned configuration and result contracts; no sklearn imports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from qr_haven.data.banknotes import DUPLICATE_POLICY

Seed = Annotated[int, Field(ge=0, le=2**32 - 1, strict=True)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Fraction = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetConfig(StrictConfig):
    path: Path = Path("data/raw/banknote_authentication/data_banknote_authentication.txt")
    reference_manifest: Path = Path("configs/datasets/banknote_authentication.json")
    expected_raw_rows: Annotated[int, Field(gt=0, strict=True)] = 1372
    duplicate_policy: Literal["exact_features_keep_first_v1"] = DUPLICATE_POLICY
    feature_order: tuple[
        Literal["variance"], Literal["skewness"], Literal["kurtosis"], Literal["entropy"]
    ] = ("variance", "skewness", "kurtosis", "entropy")
    label_mapping_status: Literal["unverified"] = "unverified"


class SplitConfig(StrictConfig):
    test_size: Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)] = 0.20
    seed: Seed = 42
    cv_folds: Annotated[int, Field(ge=2, strict=True)] = 5
    cv_seed: Seed = 43


class TrainingConfig(StrictConfig):
    estimator_seed: Seed = 44
    selection_metric: Literal["balanced_accuracy"] = "balanced_accuracy"
    tie_break_metric: Literal["f1_macro"] = "f1_macro"
    n_jobs: Literal[1] = 1
    probability_calibration: Literal[False] = False


class EvaluationConfig(StrictConfig):
    positive_label: Literal[1] = 1
    minimum_accuracy: Fraction = 0.90
    minimum_balanced_accuracy: Fraction = 0.90
    minimum_class_recall: Fraction = 0.85
    minimum_accuracy_gain: Fraction = 0.10


class LogisticConfig(StrictConfig):
    scaler: Literal["StandardScaler"] = "StandardScaler"
    solver: Literal["lbfgs"] = "lbfgs"
    max_iter: Annotated[int, Field(gt=0, strict=True)] = 2000
    class_weight: None = None
    C: Annotated[tuple[Positive, ...], Field(min_length=1)] = (0.1, 1.0, 10.0, 100.0)


class SVMConfig(StrictConfig):
    scaler: Literal["StandardScaler"] = "StandardScaler"
    kernel: Literal["rbf"] = "rbf"
    probability: Literal[False] = False
    class_weight: None = None
    C: Annotated[tuple[Positive, ...], Field(min_length=1)] = (0.1, 1.0, 10.0, 100.0)
    gamma: Annotated[tuple[Literal["scale"] | Positive, ...], Field(min_length=1)] = (
        "scale",
        0.01,
        0.1,
        1.0,
    )


class ForestConfig(StrictConfig):
    n_estimators: Annotated[int, Field(gt=0, strict=True)] = 300
    max_features: Literal["sqrt"] = "sqrt"
    class_weight: None = None
    max_depth: Annotated[
        tuple[Annotated[int, Field(gt=0, strict=True)] | None, ...], Field(min_length=1)
    ] = (None, 5, 10)
    min_samples_leaf: Annotated[
        tuple[Annotated[int, Field(gt=0, strict=True)], ...], Field(min_length=1)
    ] = (1, 2, 4)


class ModelsConfig(StrictConfig):
    baseline_strategy: Literal["most_frequent"] = "most_frequent"
    logistic_regression: LogisticConfig = Field(default_factory=LogisticConfig)
    svm: SVMConfig = Field(default_factory=SVMConfig)
    random_forest: ForestConfig = Field(default_factory=ForestConfig)


class ArtifactsConfig(StrictConfig):
    root: Path = Path("artifacts/classification/banknote_authentication")


class ClassificationConfig(StrictConfig):
    schema_version: Literal[1] = 1
    experiment: str = "banknote_authentication_v1"
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig)
    known_external_holdout_exposure: bool = False
    exposure_notes: str = "No external/manual evaluation reported; local history cannot prove this."

    @classmethod
    def from_yaml(cls, path: Path) -> ClassificationConfig:
        return cls.model_validate(yaml.safe_load(path.read_text()))

    def canonical_protocol(self) -> bool:
        default = ClassificationConfig()
        return (
            self.experiment == default.experiment
            and self.split == default.split
            and self.training == default.training
            and self.models == default.models
            and self.evaluation == default.evaluation
            and self.dataset.expected_raw_rows == 1372
        )


@dataclass(frozen=True)
class SplitManifest:
    development_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    validation_folds: dict[str, int]
    source_sha256: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TrainingResult:
    cv_results: pd.DataFrame
    selection: dict[str, Any]
    model_path: Path
    baseline_path: Path
    manifest: dict[str, Any]


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, Any]
    predictions: pd.DataFrame
    gates: dict[str, bool]
    report_path: Path


def require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            'Classification requires scikit-learn: pip install -e ".[research]"'
        ) from exc
