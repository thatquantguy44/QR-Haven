# Spec002: UCI Banknote Authentication Classifier

Status: banknote implementation complete; frozen benchmark passed on 2026-10-06.

Measured evidence: [research writeup](../../docs/research/classification/banknote_authentication.md).
The requirements below are the original experiment contract; subsequent sections describing
"proposed" files or "after implementation" commands are now implemented.

Created: 2026-10-06

Sequence: [implementation plan](00_PLAN.md) → this build → [volatility classification](02_VOLATILITY_FOLLOW_ON.md).

## 1. Objective

Implement a small, reproducible supervised-learning benchmark that predicts the supplied binary
banknote class from four measured image features. Compare logistic regression, an RBF support
vector machine, and random forest against a majority-class baseline. Target at least **90% test
accuracy**, supported by balanced accuracy, per-class recall, and an audit of the evaluation.

The scientific question is whether these features separate the supplied classes for unseen
observations from this benchmark population. The project demonstrates a complete training and
evaluation workflow. It does not establish performance on circulating currency, unfamiliar cameras,
new counterfeiting methods, or investment returns.

The score is a target, not an assumed result or a guarantee. Do not populate reports with expected
or example numbers presented as measurements.

## 2. User workflow and scope

A researcher must be able to:

1. Download the official data once or provide an already downloaded local copy.
2. Inspect validation results, provenance, duplicate counts, and the frozen experiment definition.
3. Train and compare the model families on development data.
4. Freeze the winner, then evaluate it and the baseline on the held-out test data.
5. Read a concise report and inspect the underlying metrics and individual predictions.
6. Reload the fitted preprocessing/model pipeline and classify new four-feature rows.

Required deliverables are a typed Python API, local CLI, versioned configuration, offline tests,
saved model, JSON/CSV artifacts, Markdown report, and a research writeup. A notebook is optional
and must call the same API rather than implement a separate experiment.

Excluded from v0: image upload and wavelet extraction, deep learning, synthetic training data,
oversampling, live fraud monitoring, probability calibration, deployment, and trading integration.

## 3. Source and licensing

The [official UCI dataset](https://archive.ics.uci.edu/dataset/267/banknote+authentication) describes
1,372 rows, four continuous features, and no missing values. Measurements come from wavelet
transforms of images of genuine and forged banknote-like specimens. This is measured specimen
data; the supplied table contains extracted features rather than an image collection.

| Item | Contract |
| --- | --- |
| Dataset | UCI Banknote Authentication, repository ID 267 |
| Citation | Lohweg, V. (2012). Banknote Authentication [Dataset]. UCI Machine Learning Repository. |
| DOI | [10.24432/C55P57](https://doi.org/10.24432/C55P57) |
| License | CC BY 4.0, as stated by UCI; retain attribution in manifests and reports |
| Published raw population | 1,372 rows before the duplicate policy is applied |
| Source file | `data_banknote_authentication.txt`, comma-separated, no header |
| Acquisition | [Official archive](https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip) |
| Alternate official endpoint | [UCI raw text](https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt) |

### Acquisition behavior

- Fetch only when explicitly requested by the CLI; imports, training, and tests never download.
- Use Python standard-library HTTPS and ZIP handling; no mandatory `ucimlrepo` dependency.
- Use a 30-second request timeout, at most two retries for transient failures, and a 5 MiB response
  limit. Extract only the named data member after checking its uncompressed size.
- Preserve original bytes under `data/raw/banknote_authentication/`. Record the actual source URL,
  retrieval time in UTC, byte count, SHA-256 of raw data and downloaded archive when applicable,
  license, and source citation in a sidecar manifest.
- Read an existing verified local cache without network access. A missing or corrupt cache causes
  an actionable error; do not substitute generated examples or a third-party mirror.
- Establish a reviewed reference raw-data hash in B1 before the first benchmark and commit it in
  `configs/datasets/banknote_authentication.json`, alongside the source and license. An archive
  hash may change when ZIP packaging changes; the raw-file hash defines the data version.
- Require that reviewed raw hash for a canonical benchmark run. Local files with different hashes
  may be inspected as alternate data, but cannot silently inherit the UCI benchmark identifier.
- Never overwrite an existing snapshot with different bytes. Store a new version explicitly.

The raw hash and complete class/duplicate counts are deliberately not invented in this spec;
compute them from the full download during implementation.

## 4. Canonical data contract

| Position | Canonical name | Type | Meaning |
| --- | --- | --- | --- |
| 1 | `variance` | float64 | Variance of the wavelet-transformed image |
| 2 | `skewness` | float64 | Skewness of the wavelet-transformed image |
| 3 | `kurtosis` | float64 | Kurtosis of the wavelet-transformed image; UCI metadata spells this `curtosis` |
| 4 | `entropy` | float64 | Entropy of the image |
| 5 | `target` | integer | Original source class code, exactly 0 or 1 |

Feature order is always `variance, skewness, kurtosis, entropy`. Metadata, row IDs, source position,
and `target` must never enter the feature matrix. Negative feature values are valid.

### Class-code semantics

Retain numeric labels exactly as supplied. The UCI page and inspected raw text identify a binary
class but do not establish which numeric code means genuine versus forged. Default display names
are therefore `class_0` and `class_1`; define label 1 as the statistical positive class only.

B1 should seek authoritative documentation from UCI or the dataset creator for the semantic
mapping and record its exact source if found. Until then, emit `label_mapping_status: unverified`
and use numeric class names in the report and predictions. Lack of semantic mapping does not
prevent a valid numeric classification benchmark, but it prevents claims such as "forged-note
recall" or "probability this note is counterfeit." Do not guess the mapping from feature values,
class counts, or a tutorial.

### Validation and audit

The canonical loader must:

- Parse every nonempty record as exactly five fields; reject malformed records with line numbers.
- Reject absent values, nonnumeric values, NaN, infinity, and labels outside `{0, 1}`. Do not
  silently drop invalid rows, impute values, or coerce fractional labels to integers.
- Require both classes, exactly four features, and 1,372 raw rows in canonical benchmark mode.
- Report raw counts, missingness, feature min/max, class counts, duplicate groups, and final counts.
- Treat observed source-range differences in inference as diagnostics, not automatic clipping.
- Preserve source line numbers for traceability. Keep full-population descriptive feature summaries
  out of model-selection decisions; model-development EDA uses only the development partition.

### Duplicate policy

The [official raw file](https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt)
contains repeated records. Apply one predetermined policy before any split:

1. Group records with exactly equal parsed four-feature tuples, normalizing signed zero.
2. If one tuple has multiple target labels, fail with a conflicting-label diagnostic.
3. Otherwise retain one representative, chosen by earliest source line, and preserve every source
   line in a separate lineage table. Give each retained sample equal weight.
4. Generate `sample_id` as SHA-256 of the four normalized IEEE-754 float64 values in big-endian
   feature order. Exclude the target from that identity and sort canonical samples by this ID.
5. Persist the raw-to-canonical mapping and duplicate-removal totals in the data audit.

The test denominator is the retained unique population, not automatically 1,372. Exact deduplication
does not prove independence between near-identical measurements or specimens. The supplied data
do not provide usable specimen, camera, session, or time identifiers for a stronger group split.
Record that limitation; do not apply outcome-driven rounding or near-duplicate removal.

## 5. Frozen experiment and leakage controls

### Outer split

- Use `train_test_split(test_size=0.20, stratify=y, random_state=42)` on sorted canonical samples.
- Call the 80% portion **development** and the 20% portion **test**. Development includes internal
  cross-validation training and validation folds; no third fixed validation partition is needed.
- Save actual sample membership, counts, class proportions, deduplication version, seed, and
  source hash. Use the saved membership for every candidate and every exact replay.
- Require disjoint IDs, complete coverage of retained samples, and both classes in each partition.
- Once the protocol is frozen, test labels are available only to final evaluation. Do not use test
  metrics, errors, feature distributions, or predicted scores to select models or thresholds.

### Internal validation

Use `StratifiedKFold(n_splits=5, shuffle=True, random_state=43)` within development. Persist each
sample's validation-fold assignment. Each candidate sees the same five folds and each sample is
predicted out of fold once per candidate. Fail if there are too few examples per class.

For logistic regression and SVM, put `StandardScaler` inside a scikit-learn `Pipeline`. Fit the
entire pipeline separately on each fold's training rows, then refit the selected pipeline on all
development rows. Random forest receives the original four features. Pipelines prevent learned
scaling from crossing validation boundaries, following [scikit-learn's leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).

No feature selection, PCA, resampling, hyperparameter tuning, calibration, or threshold selection
may use the test set. These extensions are excluded from the v0 search rather than left implicit.

## 6. Models and bounded search

Use scikit-learn from the existing `research` extra. Keep all settings below in the resolved
experiment configuration and log actual library versions.

| Model | Fixed settings | Search grid |
| --- | --- | --- |
| Majority baseline | `DummyClassifier(strategy="most_frequent")` | None |
| Logistic regression | `StandardScaler` → `LogisticRegression(solver="lbfgs", max_iter=2000, class_weight=None)` | `C: [0.1, 1.0, 10.0, 100.0]` |
| RBF SVM | `StandardScaler` → `SVC(kernel="rbf", probability=False, class_weight=None)` | `C: [0.1, 1.0, 10.0, 100.0]`; `gamma: ["scale", 0.01, 0.1, 1.0]` |
| Random forest | `RandomForestClassifier(n_estimators=300, max_features="sqrt", class_weight=None, random_state=44, n_jobs=1)` | `max_depth: [None, 5, 10]`; `min_samples_leaf: [1, 2, 4]` |

This gives 29 learned-model candidates and 145 cross-validation fits, plus five dummy fits and
the final winner/baseline refits. Keep the default search single-threaded and CPU-only. Log elapsed
time and machine details; no unmeasured runtime promise is part of acceptance.

Record convergence warnings and fit failures. A candidate with a failed or nonconverged fold is
ineligible; report the cause rather than quietly averaging fewer folds. If every learned candidate
fails, terminate training with diagnostics and no selected model.

### Selection rule

Rank candidates by unrounded mean validation balanced accuracy, then mean validation macro F1.
For exact ties, prefer logistic regression, then SVM, then random forest; within a family use the
declared grid order. Candidate IDs and enumeration order must be stable.

Balanced accuracy weights the two class recalls equally, as defined in
[scikit-learn's metric reference](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html).
Still report ordinary accuracy because it is the user's explicit benchmark target.

Refit only the chosen learned model and the dummy on all development data. Persist their fitted
state, feature schema, selected hyperparameters, and selection record before final evaluation.
Cross-validation scores used for model selection are development evidence, not unbiased final
generalization estimates.

### Prediction scores

- Class predictions use the fitted estimator's native `predict` behavior; do not tune a cutoff.
- For SVM, expose the label-1-oriented `decision_function` as `score_kind: decision_margin`.
- For logistic regression, forest, and dummy, use the `predict_proba` column whose `classes_` entry
  is 1, with `score_kind: probability_class_1`. Do not assume a column order without checking it.
- Do not apply a sigmoid to an SVM margin and label the result a probability. SVC probability
  estimation is disabled in v0; its behavior differs from decision scores, as documented in the
  [SVC reference](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).
- Scores support ranking metrics. No probability-calibration quality claim is made for v0.

## 7. Evaluation and target attainment

Evaluate only the frozen winner and the dummy on test rows. Candidate comparisons belong to the
development results; do not publish a test leaderboard and retrospectively pick its winner.

### Required metrics

| Metric | Reporting contract |
| --- | --- |
| Accuracy | Correct predictions / test observations, including both integer counts |
| Balanced accuracy | Mean recall across classes 0 and 1 |
| Precision, recall, F1, support | Separate values for each class and macro F1 |
| Confusion matrix | Integer counts and row-normalized version; rows = true, columns = predicted, order `[0, 1]` |
| ROC-AUC | Computed from label-1 scores, never hard predictions |
| Average precision | Label-1 scores; name it average precision, not trapezoidal PR-AUC |
| Baseline improvement | Winner accuracy minus fitted-majority-baseline test accuracy, in percentage points |
| Accuracy uncertainty | Two-sided 95% Wilson interval, with numerator and denominator |

For zero predicted support, use precision/F1 zero with an explicit diagnostic rather than suppressing
the issue. Canonical test data must contain both classes. An alternate one-class input can produce
predictions but must not produce a canonical benchmark result; undefined ranking metrics are
JSON `null` with a reason, never nonstandard JSON NaN.

Use `z = 1.959963984540054`, `p = correct / n`, and the following Wilson formula:

```text
denominator = 1 + z^2 / n
center = (p + z^2 / (2n)) / denominator
half_width = z * sqrt(p(1-p)/n + z^2/(4n^2)) / denominator
interval = [center - half_width, center + half_width]
```

This interval describes binomial sampling uncertainty under an independence assumption. It does
not cover dataset shift, unknown specimen grouping, model-selection bias from later reuse of the
test set, or all sources of measurement dependence.

### Numerical gates

| Gate | Requirement on the same untouched test set |
| --- | --- |
| User target | Accuracy ≥ 0.90 |
| Class balance guardrail | Balanced accuracy ≥ 0.90 |
| Per-class guardrail | Recall for each class ≥ 0.85 |
| Baseline guardrail | Accuracy improvement ≥ 0.10, i.e. 10 percentage points |

Use unrounded values for pass/fail. All gates and leakage/audit requirements must pass for
`benchmark_status: passed`. A 90% point estimate does not mean that the confidence interval's
lower bound exceeds 90%; report those separately. The lower bound is not an additional hidden gate.

If a gate fails, write `benchmark_status: target_not_met` and retain all artifacts. This is an
informative result, not an excuse to adjust the test split. Diagnose implementation errors using
development data and fixtures. If a correction follows exposure to test outcomes, label subsequent
scores as reused-holdout evidence; they cannot be represented as a fresh confirmatory test.

## 8. Reproducibility and experiment state

Use one manifest with states `prepared`, `trained`, `evaluated`, and `invalid`. State transitions
occur only after the corresponding artifacts have been written successfully.

- Record a protocol ID from raw-data hash, schema/duplicate-policy versions, split membership,
  search configuration, and code revision. Also record whether the worktree was dirty; for a dirty
  tree capture a hash of relevant source files rather than implying the commit alone is sufficient.
- Save Python, NumPy, pandas, scikit-learn, and QR Haven versions; resolved config; code revision;
  platform; timestamps; seeds; and artifact checksums. Save a package-version snapshot for replay.
- Hash and freeze the trained selection record and model before test evaluation.
- Require the same data, split, feature schema, configuration, and model hashes at evaluation.
- An already evaluated run returns its saved result after integrity checks. It must not overwrite
  evidence or silently retrain. A new exploratory run has a new ID and records any prior exposure
  of its holdout labels; a new directory alone does not make the test set independent again.
- Append evaluation history under the artifact root keyed by source hash and test sample IDs.
  The software can record local exposure; the model card must also disclose known external/manual
  exposure. This is a research control, not a claim that a local file can prevent all misuse.
- Require repeatable split IDs and predictions in a pinned environment. Floating-point scores
  may use a documented numerical tolerance; timings and serialized bytes need not be identical.

No all-data production refit is included in v0. The saved evaluation model remains trained only
on development data, so its test result and inference behavior refer to the same fitted object.

## 9. Repository architecture and API

The following are proposed files to create during implementation:

| Path | Responsibility |
| --- | --- |
| `src/qr_haven/data/banknotes.py` | Acquisition, parsing, validation, deduplication, and dataset manifest |
| `src/qr_haven/ml/classification/contracts.py` | Typed configuration, dataset/split references, results, and schema versions |
| `src/qr_haven/ml/classification/splits.py` | Fixed banknote split and CV membership; future time-aware policy added separately |
| `src/qr_haven/ml/classification/models.py` | Estimator factories and explicit search grids |
| `src/qr_haven/ml/classification/training.py` | Development CV, selection, and final development refit |
| `src/qr_haven/ml/classification/evaluation.py` | Metrics, uncertainty, baseline comparison, and gate evaluation |
| `src/qr_haven/ml/classification/artifacts.py` | Manifest state, persistence, integrity, exposure history, and reload |
| `src/qr_haven/ml/classification/reporting.py` | Markdown tables, model card, and CSV/JSON export |
| `src/qr_haven/ml/classification/__init__.py` | Public classification API |
| `src/qr_haven/ml/classification/__main__.py` | CLI orchestration with no separate modeling logic |
| `configs/banknote_classification.yaml` | Complete canonical protocol and output settings |
| `configs/datasets/banknote_authentication.json` | Reviewed reference raw-data hash, source identity, and license |
| `docs/research/classification/banknote_authentication.md` | Research writeup and measured result links |
| `docs/api/classification.md` | Public API, artifact schemas, CLI, and inference examples |
| `tests/test_banknote_data.py` | Parser, provenance, and duplicate-policy checks |
| `tests/test_classification.py` | Split, pipeline, selection, metrics, CLI, and persistence checks |

Existing `qr_haven.data` price portals are market-specific; the banknote loader must not pretend
these image statistics are prices. Preserve existing imports, including the NumPy LSTM in
`qr_haven.ml`. Do not introduce eager scikit-learn imports into the base `qr_haven` or `qr_haven.ml`
import path. Missing optional dependencies should explain how to install `.[research]`.

Public API contract, with concrete dataclasses/types to be implemented:

```python
fetch_banknote_dataset(cache_dir: Path) -> Path
load_banknote_dataset(path: Path, *, reference_sha256: str) -> BanknoteDataset
prepare_banknote_split(dataset: BanknoteDataset, config: ClassificationConfig) -> SplitManifest
train_banknote_classifier(dataset: BanknoteDataset, split: SplitManifest,
                         config: ClassificationConfig, output_dir: Path) -> TrainingResult
evaluate_banknote_run(run_dir: Path, dataset: BanknoteDataset) -> EvaluationResult
load_classifier(run_dir: Path) -> ClassificationModel
# ClassificationModel.predict(features: pd.DataFrame) -> pd.DataFrame
```

`BanknoteDataset` contains the canonical feature frame, aligned target series, stable IDs,
lineage, source manifest, and validation audit. `SplitManifest` contains development/test IDs and
development fold assignments. `TrainingResult` contains the CV result table, selection, fitted
artifact paths, baseline, and run manifest. `EvaluationResult` contains metrics, predictions,
gate results, and report paths.

Inference accepts named columns and reorders them to the saved schema. Reject missing, duplicate,
or extra feature columns, nonfinite values, and a supplied target column. Preserve input row order
and index; prediction output contains `predicted_class`, `score_class_1`, and `score_kind`.
Semantic names are optional only when source-verified. Inference never refits or fetches data.

## 10. Configuration and CLI

The initial configuration must encode the following fixed protocol values in addition to all
model settings from section 6:

```yaml
schema_version: 1
experiment: banknote_authentication_v1
dataset:
  path: data/raw/banknote_authentication/data_banknote_authentication.txt
  reference_manifest: configs/datasets/banknote_authentication.json
  expected_raw_rows: 1372
  duplicate_policy: exact_features_keep_first_v1
  feature_order: [variance, skewness, kurtosis, entropy]
  label_mapping_status: unverified
split:
  test_size: 0.20
  seed: 42
  cv_folds: 5
  cv_seed: 43
training:
  estimator_seed: 44
  selection_metric: balanced_accuracy
  tie_break_metric: f1_macro
  n_jobs: 1
  probability_calibration: false
evaluation:
  positive_label: 1
  minimum_accuracy: 0.90
  minimum_balanced_accuracy: 0.90
  minimum_class_recall: 0.85
  minimum_accuracy_gain: 0.10
artifacts:
  root: artifacts/classification/banknote_authentication
```

This is a configuration contract, not a file created by this specification change. The implemented
config must also contain explicit model grids and validate unknown keys, invalid seeds/fractions,
and incompatible schema versions. The B1 reference manifest supplies the reviewed raw-data hash.

Expected commands after implementation:

```bash
python -m pip install -e ".[dev,research]"

python -m qr_haven.ml.classification fetch-banknotes \
  --cache-dir data/raw/banknote_authentication

python -m qr_haven.ml.classification train \
  --config configs/banknote_classification.yaml \
  --run-id banknote-v1

python -m qr_haven.ml.classification evaluate \
  --run-dir artifacts/classification/banknote_authentication/banknote-v1

python -m qr_haven.ml.classification predict \
  --run-dir artifacts/classification/banknote_authentication/banknote-v1 \
  --input path/to/features.csv \
  --output path/to/predictions.csv
```

`train` prepares and freezes the split on the first run, records all development results, and does
not compute test performance. It prints the next evaluation command. `evaluate` uses the dataset
path recorded in the run manifest and validates its hash. `fetch-banknotes` records observed
provenance and verifies bytes against the committed reference manifest. Establishing that
reference is an implementation task in B1; ordinary users run the commands above without an
additional approval or reference-generation step. Dataset-version updates require a new reviewed
reference and protocol ID rather than silently changing an existing benchmark.

Exit codes: 0 for a successful command, 1 for an input/operational/integrity failure, and 2 for an
evaluation that completed but missed the numerical gates. Exit code 2 still produces the full
report. A noncanonical or previously exposed run may produce diagnostics but cannot exit as a
passing fresh benchmark. Print its status and reason explicitly.

Use atomic artifact writes and refuse to overwrite existing run IDs or prediction files. Emit
clear messages for missing cache, absent optional dependencies, malformed data, unknown schema,
insufficient class counts, invalid experiment state, and serialization-version mismatches.

## 11. Artifact and report contract

Each run directory must contain:

| Artifact | Required content |
| --- | --- |
| `manifest.json` | Schema version, state, protocol ID, provenance, code/environment, timestamps, checksums, exposure status |
| `config.resolved.yaml` | All defaults, exact model grids, seeds, and gates |
| `environment.txt` | Installed package versions sufficient to recreate the training environment |
| `data_audit.json` | Raw/retained counts, class counts, validation, duplicate policy, lineage reference, semantic mapping status |
| `lineage.csv` | Original source line → retained sample ID |
| `split.csv` | Sample ID, development/test role, and validation fold for development rows |
| `cv_results.csv` | Every candidate/fold, parameter values, metrics, fit time, and warnings/failures |
| `selection.json` | Ranking rule, chosen candidate, development scores, and freeze timestamp |
| `model.pkl`, `baseline.pkl` | Complete fitted winner pipeline and fitted majority baseline |
| `metrics.json` | Winner/baseline test metrics, interval, gate booleans, status, and reasons |
| `test_predictions.csv` | Sample ID, true label, predicted label, label-1 score, score kind, and baseline prediction |
| `report.md` | Human-readable model card and experiment results |

Use standard-library pickle for local persistence and load only trusted, locally produced
artifacts. Check stored environment/schema metadata before deserializing. Pickle-compatible model
formats can execute code on load and cross-version support is limited; see the
[scikit-learn persistence guidance](https://scikit-learn.org/stable/model_persistence.html).
Do not accept arbitrary uploaded model files or promise cross-version loading.

Keep generated files under the existing ignored `artifacts/` and data directories. Commit the
configuration, loader, source citation, tests, and concise research writeup, not raw downloads or
binary model bundles. Do not alter existing unrelated experiments.

The report must answer, in order:

1. What was predicted, and did the frozen benchmark meet its target?
2. Which real data version and class-code definitions were used?
3. How many rows survived validation/deduplication, and how were they split?
4. Which model won development CV, and how did the candidates compare there?
5. What were the final winner and baseline test metrics, counts, and uncertainty interval?
6. Which individual errors occurred, with source lineage, without turning that analysis into tuning?
7. What can be reproduced, and what are the population/independence/calibration limitations?
8. What is the next project? Link the queued volatility brief.

Markdown tables suffice for v0: include development comparison, both confusion matrices, per-class
metrics, and every gate. Charts and interactive dashboards are optional follow-up work.

## 12. Tests and verification

Normal tests use deterministic local fixtures and require no network. Keep their assertions about
correctness and evaluation integrity rather than achieving 90% on synthetic or tiny fixture data.

| Area | Meaningful checks |
| --- | --- |
| Data parsing | Headerless schema, valid negative inputs, malformed fields, missing/nonfinite values, bad labels, and source-row diagnostics |
| Data identity | Correct duplicate collapse and lineage, conflicting-label failure, signed-zero normalization, and hash mismatch rejection |
| Partitions | Disjoint/complete sample membership, class presence, stable reproduction, and exactly one validation assignment per development sample |
| Leakage | Instrument pipeline fitting to show held-out rows never fit scalers/models; changing test labels cannot change selection or the saved development fit |
| Selection | Hand-built fold scores exercise ranking, tie-breaking, failed-fold exclusion, and baseline separation |
| Metrics | Known prediction examples verify accuracy, balanced accuracy, per-class values, matrix orientation, score orientation, Wilson bounds, and exact gate boundaries |
| Persistence/inference | Reload reproduces predictions and scores, feature reordering is correct, invalid schemas fail, and inference does not fit |
| Experiment state | Evaluation before training fails; hash mismatches fail; repeat evaluation preserves prior evidence; exposed holdouts cannot masquerade as new benchmarks |
| CLI | Local fetch fixture/cache behavior, useful errors, exit codes, and complete artifact outputs |
| Optional dependencies | Existing base-package/ML imports still work without the `research` extra |

The real-data integration run separately verifies the reference source hash, expected raw schema,
duplicate audit, frozen protocol, report completeness, and actual numerical outcome. It is an
explicit benchmark step, not a mandatory network download in normal unit tests.

After focused tests, use the existing repository checks: `pytest`, `ruff check .`, and `mypy src`.
Record pre-existing failures separately. No frontend, browser, or deployment checks are required.

## 13. Definition of done and follow-on handoff

Engineering completion requires acquisition and offline replay, all model families, a frozen
selection/evaluation boundary, working inference, complete artifacts, documentation, and passing
relevant checks. Benchmark success additionally requires all section 7 gates and an unexposed
test evaluation. Keep those two statuses separate in the report and project plan.

When the engineering work and its benchmark report are complete, the next project is
[next-period high-volatility versus normal-volatility classification](02_VOLATILITY_FOLLOW_ON.md).
Reuse model factories, metric utilities, artifact conventions, and report structure. Replace the
random split and banknote schema with market-data features, forward labels, and purged
chronological evaluation. Begin that work only after the banknote milestones close.
