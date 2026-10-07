"""Render the spec002 model comparison from saved, verified run artifacts."""

# Matplotlib must be imported after its cache location is redirected below.
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

cache_root = Path(tempfile.gettempdir()) / "qr-haven-matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FAMILY_LABELS = {
    "baseline": "Majority baseline\nmost frequent",
    "logistic_regression": "Logistic regression\nC=100",
    "random_forest": "Random forest\ndepth=None, leaf=1",
    "svm": "RBF SVM (selected)\nC=10, gamma=scale",
}
FAMILY_COLORS = {
    "baseline": "#8A9099",
    "logistic_regression": "#C79532",
    "random_forest": "#778B62",
    "svm": "#2F6B9A",
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _development_rows(run_dir: Path) -> pd.DataFrame:
    selection = _load_json(run_dir / "selection.json")
    cv = pd.read_csv(run_dir / "cv_results.csv")
    chosen_by_family = {
        family: next(row for row in selection["ranking"] if row["family"] == family)[
            "candidate_id"
        ]
        for family in ("logistic_regression", "random_forest", "svm")
    }
    chosen_by_family["baseline"] = "baseline"
    records = []
    for family in ("baseline", "logistic_regression", "random_forest", "svm"):
        candidate = chosen_by_family[family]
        folds = cv.loc[cv["candidate_id"] == candidate, "balanced_accuracy"].astype(float)
        if len(folds) != 5:
            raise ValueError(f"Expected five CV folds for {candidate}; observed {len(folds)}")
        records.append(
            {
                "family": family,
                "candidate_id": candidate,
                "mean": float(folds.mean()),
                "minimum": float(folds.min()),
                "maximum": float(folds.max()),
                "folds": folds.tolist(),
            }
        )
    return pd.DataFrame(records)


def render(run_dir: Path, output: Path) -> None:
    development = _development_rows(run_dir)
    metrics = _load_json(run_dir / "metrics.json")

    background = "#F7F5EF"
    text = "#20242A"
    grid = "#D9D6CF"
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(15, 8), gridspec_kw={"width_ratios": [1.65, 1]}, facecolor=background
    )
    for axis in (left, right):
        axis.set_facecolor(background)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#777B81")
        axis.tick_params(colors=text, labelsize=10)
        axis.xaxis.grid(True, color=grid, linewidth=0.8, zorder=0)
        axis.set_xlim(0, 1.04)
        axis.set_xticks(np.linspace(0, 1, 6), [f"{value:.0%}" for value in np.linspace(0, 1, 6)])

    y = np.arange(len(development))
    families = development["family"].tolist()
    colors = [FAMILY_COLORS[family] for family in families]
    means = development["mean"].to_numpy()
    left.barh(y, means, height=0.54, color=colors, edgecolor=text, linewidth=0.6, zorder=2)
    left.set_yticks(y, [FAMILY_LABELS[family] for family in families])
    left.invert_yaxis()
    left.set_xlabel("Mean balanced accuracy across five development folds", color=text, labelpad=10)
    left.set_title("Development model comparison", loc="left", color=text, weight="bold", pad=16)

    rng = np.random.default_rng(42)
    for row_number, row in development.iterrows():
        jitter = rng.uniform(-0.12, 0.12, len(row["folds"]))
        left.scatter(
            row["folds"], row_number + jitter, s=28, facecolor=background,
            edgecolor=text, linewidth=0.8, zorder=3,
        )
        left.text(
            min(float(row["mean"]) + 0.012, 1.012), row_number,
            f"{row['mean']:.2%}", va="center", ha="left", color=text, weight="bold", fontsize=10,
        )

    test_names = ["Majority baseline", "Selected RBF SVM"]
    test_values = [metrics["baseline"]["accuracy"], metrics["winner"]["accuracy"]]
    test_colors = [FAMILY_COLORS["baseline"], FAMILY_COLORS["svm"]]
    test_y = np.arange(2)
    right.barh(
        test_y, test_values, height=0.54, color=test_colors, edgecolor=text, linewidth=0.6, zorder=2
    )
    right.set_yticks(test_y, test_names)
    right.invert_yaxis()
    right.set_xlabel("Accuracy on 270 frozen test samples", color=text, labelpad=10)
    right.set_title("Held-out test evaluation", loc="left", color=text, weight="bold", pad=16)
    for row_number, value in enumerate(test_values):
        count = metrics["baseline" if row_number == 0 else "winner"]["correct"]
        right.text(
            min(value + 0.018, 1.015), row_number, f"{value:.2%}\n({count}/270)",
            va="center", ha="left", color=text, weight="bold", fontsize=10,
        )

    fig.suptitle(
        "UCI Banknote Authentication — Model Comparison",
        x=0.06, y=0.965, ha="left", fontsize=19, weight="bold", color=text,
    )
    fig.text(
        0.06, 0.907,
        "Dots show individual development folds. Test performance is reported only for the "
        "preselected winner and baseline.",
        ha="left", fontsize=11, color="#555B63",
    )
    fig.text(
        0.06, 0.035,
        "Source: saved banknote-v1 artifacts. Development n=1,078; test n=270 after exact-feature "
        "deduplication. Numeric class semantics remain unverified.",
        ha="left", fontsize=9.5, color="#555B63",
    )
    fig.subplots_adjust(left=0.19, right=0.965, top=0.83, bottom=0.16, wspace=0.38)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor=background, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path,
        default=Path("artifacts/classification/banknote_authentication/banknote-v1"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/classification/banknote_authentication/verification/model_comparison.png"),
    )
    args = parser.parse_args()
    render(args.run_dir, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
