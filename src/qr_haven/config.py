"""Configuration primitives shared by QR-Haven subsystems."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QRHavenSettings:
    """Runtime paths and environment labels for research workflows."""

    project_root: Path = Path(".")
    data_dir: Path = Path("data")
    model_dir: Path = Path("models")
    experiment_dir: Path = Path("experiments")
    environment: str = "local"

