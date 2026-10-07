# Banknote classification API and CLI

Spec002 implements the [frozen banknote protocol](../../specs/spec002/01_SPEC.md).
The separate [volatility project](../../specs/spec002/03_VOLATILITY_SPEC.md) now has a frozen
dual-source data contract and is in progress.
The workflow is local, CPU-only and explicitly separates training from final evaluation.

## Install and run

From the repository root, use Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,research]"

# Explicit acquisition; an intact verified cache is read without network access.
python -m qr_haven.ml.classification fetch-banknotes \
  --cache-dir data/raw/banknote_authentication

python -m qr_haven.ml.classification train \
  --config configs/banknote_classification.yaml \
  --run-id banknote-v1

python -m qr_haven.ml.classification evaluate \
  --run-dir artifacts/classification/banknote_authentication/banknote-v1
```

To train from the supplied local copy instead of downloading, add
`--data data/input/banknote/data_banknote_authentication.txt` to `train`. Its raw byte hash
must match the reviewed [dataset reference](../../configs/datasets/banknote_authentication.json).
Training and imports never fetch data. Paths in configuration are relative to the working
directory; the run manifest stores the source's absolute path for evaluation.

Run IDs are immutable. Choose a new ID for an exploratory experiment; do not delete or rename
the evaluation history to claim another fresh result. Repeating `evaluate` on an evaluated run
checks artifact and input integrity, then returns the original result without recomputing it.

Exit codes are 0 for success, 1 for malformed inputs/operational/integrity failures, and 2 for a
completed evaluation that misses a numerical gate or cannot claim a fresh canonical benchmark.
Both target misses and exploratory evaluations retain their reports and predictions.

## Public Python API

```python
from pathlib import Path
import json
from qr_haven.ml.classification import (
    ClassificationConfig,
    load_banknote_dataset,
    prepare_banknote_split,
    train_banknote_classifier,
    evaluate_banknote_run,
    load_classifier,
)

config = ClassificationConfig.from_yaml(Path("configs/banknote_classification.yaml"))
reference = json.loads(config.dataset.reference_manifest.read_text())
dataset = load_banknote_dataset(
    config.dataset.path, reference_sha256=reference["raw_sha256"]
)
split = prepare_banknote_split(dataset, config)
run_dir = config.artifacts.root / "my-experiment"
trained = train_banknote_classifier(dataset, split, config, run_dir)
# No test metrics have been computed at this point.
evaluated = evaluate_banknote_run(run_dir, dataset)
model = load_classifier(run_dir)
```

| Type | Contents |
| --- | --- |
| `BanknoteDataset` | Canonical float64 features, aligned integer targets, sorted stable IDs, lineage, source manifest and audit |
| `ClassificationConfig` | Strict nested configuration, explicit grids, split seeds, gates and exposure disclosure |
| `SplitManifest` | Ordered development/test membership, validation-fold assignments, source hash, class counts/proportions |
| `TrainingResult` | Every candidate/fold result, frozen selection, winner/baseline paths and run manifest |
| `EvaluationResult` | Winner/baseline metrics, individual predictions, gate booleans and report path |
| `ClassificationModel` | Fitted estimator/pipeline, saved feature order, source-range diagnostics and `predict` |

To inspect alternate data, call `load_banknote_dataset(path, reference_sha256=None,
canonical=False)`. The same strict parser and duplicate policy apply. Alternate datasets and
modified protocols cannot receive `benchmark_status: passed`. Canonical loading requires both
classes, 1,372 raw rows, and the reviewed raw checksum. No semantic mapping of class codes is
inferred from a score or a tutorial.

## Inference

Feature CSVs must have exactly these named columns (any order):

```csv
variance,skewness,kurtosis,entropy
3.6216,8.6661,-2.8073,-0.44699
```

```bash
python -m qr_haven.ml.classification predict \
  --run-dir artifacts/classification/banknote_authentication/banknote-v1 \
  --input path/to/features.csv --output path/to/predictions.csv
```

The Python API preserves the caller's row order and index, including repeated index values:

```python
import pandas as pd
predictions = model.predict(pd.read_csv("path/to/features.csv"))
print(predictions)
print(predictions.attrs["out_of_source_range_counts"])
```

Output columns are `predicted_class`, `score_class_1`, and `score_kind`. SVM scores are
`decision_margin`; logistic, forest and dummy scores are `probability_class_1`. The probability
column is selected from the estimator's `classes_`; probabilities are not claimed to be calibrated.
Numeric class 1 is the statistical positive class, with an unverified genuine/forged mapping.

Missing/extra/duplicate columns, a target column and nonfinite inputs are rejected. Values outside
source ranges are counted in diagnostics, without clipping. Inference never fits or fetches.
The CLI refuses to overwrite an existing prediction file.

## Frozen experiment and artifacts

The 29 learned candidates share the same five development folds. A majority-class baseline
adds five CV fits. Logistic regression and SVM fit their `StandardScaler` inside each fold's
pipeline; random forest uses unscaled features. Only the winner and baseline are refitted on
all development rows. Failed or nonconverged folds make a candidate ineligible.

Selection uses unrounded mean balanced accuracy, then macro F1, then logistic/SVM/forest order
and the declared parameter-grid order. The test set is never a candidate leaderboard.

Generated files live under the ignored artifact root:

| File | Contract |
| --- | --- |
| `manifest.json` | State, protocol hash, source/code/environment/machine identity, checksums, exposure and timestamps |
| `config.resolved.yaml` | All settings and defaults actually used |
| `environment.txt` | Installed distribution versions for replay |
| `source_snapshot.zip` | The exact classification/loader source files, including dirty-tree changes |
| `data_audit.json`, `lineage.csv` | Validation, duplicate groups, class counts, source ranges and every source line |
| `split.json`, `split.csv` | Ordered split membership, fold assignments and class proportions |
| `cv_results.csv` | Every candidate/fold, parameters, accuracy, balanced accuracy, macro F1, time and failure/warning text |
| `cv_predictions.csv` | One out-of-fold prediction per development sample for each successful candidate |
| `selection.json` | Complete ranking, chosen parameters, actual estimator settings and freeze timestamp |
| `model.pkl`, `baseline.pkl` | Complete fitted development-only model bundles |
| `metrics.json` | Both test metric sets, Wilson interval, gates, audit checks and benchmark status |
| `test_predictions.csv` | Sample ID, true/predicted labels, winner/baseline scores and source lines |
| `report.md` | Model card, development comparison, test metrics, individual errors and limitations |

State transitions are `prepared` → `trained` → `evaluated`; failed training or an interrupted
evaluation after exposure leaves an `invalid` run. Files are atomically published and a state
transition is written only after its artifacts. Integrity checks bind the resolved configuration,
saved membership, selected model and original data. An evaluation reservation is appended to
`evaluation_history.jsonl` under the artifact root before test labels are used. Overlapping test
IDs for the same source are treated as previously exposed, including a changed split that overlaps.
Concurrent evaluations under one root are serialized with a POSIX file lock.

Set `known_external_holdout_exposure: true` and explain it in `exposure_notes` if test outcomes
were already inspected elsewhere. Local history cannot detect deliberate ledger removal, another
artifact root, or unreported external use. No software control establishes specimen independence.

Only load trusted, locally produced pickle bundles. Checksums detect corruption, not malicious
serialization. Loading checks the recorded Python/library versions, artifact schema and relevant
classification source hash before deserializing. Restore the original environment/source to
replay a saved model. Scores can be compared with absolute tolerance `1e-12`; elapsed times and
pickle bytes need not match. The saved model is never refitted on all data.
The relevant source snapshot is archived under package-relative paths; restore it only into
a trusted checkout of the recorded repository revision, not an unrelated directory.

## Verification

```bash
python -m pytest tests/test_banknote_data.py tests/test_classification.py -q
python -m pytest -q
python -m ruff check .
python -m mypy src
```

Ordinary tests use local deterministic fixtures and mock acquisition. The real-data run is a
separate integration benchmark. A missed research target is preserved as a result, not disguised
as a software test failure. The measured outcome and any pre-existing repository issues are
recorded in the [research writeup](../research/classification/banknote_authentication.md).
