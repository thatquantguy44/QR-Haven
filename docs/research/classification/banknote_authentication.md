# UCI Banknote Authentication: frozen classification benchmark

Completed 2026-10-06 (America/New_York). **Engineering: complete. Benchmark: passed.**

The selected RBF SVM correctly classified **270/270 untouched test samples (100.00%)**.
Balanced accuracy and both class recalls were also 100.00%. The majority baseline scored
148/270 (54.81%), so the improvement was **45.19 percentage points**. The two-sided 95% Wilson
accuracy interval was **98.60%–100.00%**, assuming independent observations.

This predicts the numeric class supplied with four image statistics. It does not establish
performance on circulating currency, unseen cameras, new counterfeit methods or financial returns.

## Evidence and reproduction

Run ID: `banknote-v1`. Generated artifacts remain in the ignored local artifact directory.

- [Full model card](../../../artifacts/classification/banknote_authentication/banknote-v1/report.md)
- [Final metrics and gates](../../../artifacts/classification/banknote_authentication/banknote-v1/metrics.json)
- [Individual test predictions](../../../artifacts/classification/banknote_authentication/banknote-v1/test_predictions.csv)
- [Frozen selection](../../../artifacts/classification/banknote_authentication/banknote-v1/selection.json)
- [Every development candidate/fold](../../../artifacts/classification/banknote_authentication/banknote-v1/cv_results.csv)
- [Protocol, environment and checksums](../../../artifacts/classification/banknote_authentication/banknote-v1/manifest.json)
- [Integration verification](../../../artifacts/classification/banknote_authentication/verification/integration_checks.json)
- [API, CLI and artifact documentation](../../api/classification.md)

The tracked [configuration](../../../configs/banknote_classification.yaml),
[reviewed source reference](../../../configs/datasets/banknote_authentication.json), source code
and tests reproduce the procedure. Restore the recorded source/environment for exact model
replay. The generated run also contains `environment.txt` and `source_snapshot.zip`; the latter
preserves relevant source files from the dirty working tree used for this experiment.

```bash
source .venv/bin/activate
python -m qr_haven.ml.classification evaluate \
  --run-dir artifacts/classification/banknote_authentication/banknote-v1
```

That command verifies and returns the existing result. For a new development run using the
supplied input, choose a new run ID and follow the CLI guide with
`--data data/input/banknote/data_banknote_authentication.txt`. Evaluation under the same artifact
root will identify the already exposed holdout as exploratory, even if numerical gates pass.
Replaying a split does not create new confirmatory evidence.

## Source, validation and lineage

[UCI Banknote Authentication](https://archive.ics.uci.edu/dataset/267/banknote+authentication),
repository ID 267. Citation: Lohweg, V. (2012). Banknote Authentication [Dataset]. UCI Machine
Learning Repository. [DOI 10.24432/C55P57](https://doi.org/10.24432/C55P57). License: CC BY 4.0.

The supplied `data/input/banknote/data_banknote_authentication.txt` was compared with the named
data member downloaded from the official UCI archive. The bytes matched exactly:

| Item | Observed value |
| --- | --- |
| Raw bytes | 46,400 |
| Raw SHA-256 | `d0539aaed2139ba7a587b3e34fb345ce503ff7d5d33dbf9912d8e195ce425cb9` |
| Archive SHA-256 | `1e2acd9a2085fadf3d8145c12d3d22af853320d52294a6590c2eaf75fdc05227` |
| Retrieval UTC | 2026-10-07T01:21:11.672197+00:00 (October 6 in New York) |
| Raw rows | 1,372: class_0 = 762; class_1 = 610 |
| Missing/nonfinite fields | 0 |
| Duplicate groups | 11 |
| Repeated rows removed | 24, all class_0 |
| Unique samples retained | 1,348: class_0 = 738; class_1 = 610 |

The absence of a terminal newline makes `wc -l` display 1,371; the strict record parser verifies
1,372 records. Exact parsed four-feature duplicates were collapsed before splitting, normalizing
signed zero and retaining the earliest source line. Stable sample IDs hash normalized big-endian
IEEE-754 float64 features, excluding the target. Every original line is retained in the
[lineage table](../../../artifacts/classification/banknote_authentication/banknote-v1/lineage.csv).

UCI describes genuine and forged specimens but does not establish their numeric-code mapping
in its metadata. No authoritative mapping was established during this build. The report retains
`class_0` and `class_1`, with `label_mapping_status: unverified`. Label 1 is the statistical positive
class. No metric is labelled counterfeit recall or counterfeit probability.

## Frozen experiment

Canonical samples were sorted by sample ID, then split with stratification, `test_size=0.20`
and seed 42. Development folds use shuffled five-fold stratification with seed 43.

| Partition | Samples | class_0 | class_1 |
| --- | ---: | ---: | ---: |
| Development | 1,078 | 590 | 488 |
| Test | 270 | 148 | 122 |

All 29 learned candidates used identical saved development folds. The majority baseline added
five CV fits, for **150 successful CV fits** and **32,340 out-of-fold predictions**. Only the
chosen candidate and the baseline were refitted on all development rows. Scaling for logistic
regression and SVM was fitted separately within each development fold. No test results, scores
or distributions selected the winner or its threshold.

Selection used unrounded mean balanced accuracy, then macro F1, then declared family/grid order.
The best development candidate in each family was:

| Model | Parameters | Mean accuracy | Mean balanced accuracy | Mean macro F1 |
| --- | --- | ---: | ---: | ---: |
| Logistic regression | C = 100 | 99.7209% | 99.7274% | 0.997183 |
| **RBF SVM (selected)** | **C = 10; gamma = scale** | **100.0000%** | **100.0000%** | **1.000000** |
| Random forest | 300 trees; max_depth = None; min_samples_leaf = 1 | 99.6288% | 99.6610% | 0.996260 |

Candidate ID: `svm_08`. Development CV is model-selection evidence, not an unbiased final estimate.
The complete grid, including other ties, is retained in `selection.json` and `cv_results.csv`.
No candidate failed or raised a convergence warning. Scikit-learn 1.9.1 emitted a deprecation
warning for the spec's explicit `SVC(probability=False)` parameter in 80 SVM fold fits and the
final SVM fit; these warnings are recorded. Calibration remains disabled as specified.

Training took 7.634 seconds on the recorded machine, with estimator and native-library thread
limits of one. This is a measurement of this run, not a runtime guarantee. Environment:
Python 3.12.0, NumPy 2.4.6, pandas 3.0.3, SciPy 1.18.1, scikit-learn 1.9.1, QR Haven 0.1.0.

## Final evaluation

The model and selection were saved and checksummed while the run was `trained`, before final
evaluation. There was one local evaluation reservation and no reported external/manual evaluation.
The test set was evaluated once on October 6, 2026, at 21:30:38 America/New_York.

| Metric | Frozen winner | Majority baseline |
| --- | ---: | ---: |
| Correct / total | 270 / 270 | 148 / 270 |
| Accuracy | 100.00% | 54.81% |
| Balanced accuracy | 100.00% | 50.00% |
| Macro F1 | 1.000000 | 0.354067 |
| ROC-AUC | 1.000000 | 0.500000 |
| Average precision | 1.000000 | 0.451852 |
| Accuracy Wilson 95% interval | 98.60%–100.00% | 48.85%–60.64% |

The SVM exposes native label-1 decision margins, not probabilities. Ranking metrics use those
margins. Baseline scores are label-1 probabilities from the fitted majority classifier.

| Model / class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Winner / class_0 | 1.000000 | 1.000000 | 1.000000 | 148 |
| Winner / class_1 | 1.000000 | 1.000000 | 1.000000 | 122 |
| Baseline / class_0 | 0.548148 | 1.000000 | 0.708134 | 148 |
| Baseline / class_1 | 0.000000 | 0.000000 | 0.000000 | 122 |

Winner confusion matrix (true rows, predicted columns, order 0/1): `[[148, 0], [0, 122]]`;
row-normalized: `[[1, 0], [0, 1]]`. Baseline: `[[148, 0], [122, 0]]`;
row-normalized: `[[1, 0], [1, 0]]`. The baseline's class_1 precision/F1 are zero because it never
predicts that class; the metric output explicitly records this diagnostic.

| Gate | Requirement | Measured | Outcome |
| --- | ---: | ---: | --- |
| Accuracy | ≥ 90% | 100% | Pass |
| Balanced accuracy | ≥ 90% | 100% | Pass |
| class_0 recall | ≥ 85% | 100% | Pass |
| class_1 recall | ≥ 85% | 100% | Pass |
| Accuracy improvement | ≥ 10 percentage points | 45.185185 percentage points | Pass |

All gates use unrounded values. There were **no individual winner test errors**. Individual
predictions still carry sample IDs and source-line lineage for independent inspection.

## Engineering verification and limits

- **807 repository tests passed**, including 51 new data/classification tests, with 91 warnings.
- New source and tests pass focused Ruff checks. The 10 new source modules pass focused mypy.
- `ruff check .` still reports **184 pre-existing findings**. `mypy src` reports **65 errors in
  17 existing files**, with none in the new modules. The initial environment had 67 type errors;
  installing missing optional dependencies removed two missing-import findings.
- The initial system environment lacked scikit-learn and gpytorch. A project-local `.venv`
  supplies the declared research dependencies and gpytorch for the existing borrow-demand tests;
  no unrelated model code was changed to make those tests pass.
- Integrity checks verified 150 complete CV fits, one OOF prediction per candidate/development
  sample, no OOF/test overlap, saved-model prediction identity, score agreement within `1e-12`,
  and byte-identical artifacts/history after repeat evaluation. CLI inference also succeeded
  with reordered feature columns.

Check logs are retained under
[verification](../../../artifacts/classification/banknote_authentication/verification/integration_checks.json).
The [tests](../../../tests/test_classification.py) additionally instrument model/scaler fits,
alter held-out labels to demonstrate selection independence, exercise exact gate boundaries,
and check invalid states, corruption, interrupted evaluation and prior exposure.

Exact duplicates do not capture all possible specimen dependence. The dataset lacks specimen,
camera, session and time identifiers for stronger grouping. Near-duplicates were not removed
using outcome-driven thresholds. The Wilson interval assumes independent binomial outcomes and
does not cover dataset shift, unknown grouping or later tuning against these test labels.
No calibration-quality claim is made. The saved model remains fitted only on development data.
The exposure ledger describes this artifact root; it cannot prove the absence of unreported
manual or external use.

## Handoff

Banknote milestones B1–B5 are closed. The next project is the separate
[next-five-session high/normal-volatility classifier](../../../specs/spec002/02_VOLATILITY_FOLLOW_ON.md).
Its market-data provider, adjusted-price snapshot and full implementation contract must be fixed
before fitting. It remains queued; the banknote result supplies no evidence of market predictability.
