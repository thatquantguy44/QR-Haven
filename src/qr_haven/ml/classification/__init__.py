"""Local reproducible classification API; sklearn is loaded only when needed."""

from qr_haven.data.banknotes import BanknoteDataset, fetch_banknote_dataset, load_banknote_dataset
from qr_haven.ml.classification.artifacts import load_classifier
from qr_haven.ml.classification.contracts import (
    ClassificationConfig,
    EvaluationResult,
    SplitManifest,
    TrainingResult,
)
from qr_haven.ml.classification.evaluation import evaluate_banknote_run
from qr_haven.ml.classification.models import ClassificationModel
from qr_haven.ml.classification.splits import prepare_banknote_split
from qr_haven.ml.classification.training import train_banknote_classifier

__all__ = [
    "BanknoteDataset",
    "ClassificationConfig",
    "ClassificationModel",
    "EvaluationResult",
    "SplitManifest",
    "TrainingResult",
    "evaluate_banknote_run",
    "fetch_banknote_dataset",
    "load_banknote_dataset",
    "load_classifier",
    "prepare_banknote_split",
    "train_banknote_classifier",
]
