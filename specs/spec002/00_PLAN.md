# Spec002 Plan: Supervised Finance Classification

Status: banknote milestones B1–B5 complete; benchmark passed. Volatility V1–V2 are complete;
V3 model comparison is next.

Created: 2026-10-06

## Objective and sequence

Build a reproducible classifier on the real UCI Banknote Authentication dataset, targeting at
least 90% accuracy on an untouched test partition. Then work on forecasting next-period high
volatility versus normal volatility using historical market data.

The banknote benchmark establishes data validation, model comparison, evaluation, and artifact
contracts. The volatility project reuses those contracts with a separate chronological validation
policy. Its success criterion is improvement over realistic baselines, not a promised 90% score.

## Documents

| Document | Purpose |
| --- | --- |
| [01_SPEC.md](01_SPEC.md) | Full banknote data, training, evaluation, API, CLI, artifact, and acceptance specification. |
| [02_VOLATILITY_FOLLOW_ON.md](02_VOLATILITY_FOLLOW_ON.md) | Queued next project, target definition, validation requirements, and implementation sequence. |
| [03_VOLATILITY_SPEC.md](03_VOLATILITY_SPEC.md) | Frozen data profiles, timing, model grid, validation, artifacts, and acceptance contract. |

The banknote commands, APIs, configuration and outputs are implemented. See the
[measured research writeup](../../docs/research/classification/banknote_authentication.md) and
[API guide](../../docs/api/classification.md). The volatility data layer now supports a validated
local SPX profile and the recommended Tiingo SPY profile; feature and modeling work remains.

## Implementation milestones

| Milestone | Work | Exit evidence |
| --- | --- | --- |
| B1: Data contract | Implement acquisition, local loading, schema checks, provenance, exact-duplicate handling, and stable sample IDs. | Offline parser/validation tests pass; a real-data audit records source hashes, counts, and class-code semantics. |
| B2: Frozen experiment | Create the deduplicated 80/20 split and five development folds; serialize membership and configuration. | No ID overlaps development/test or a fold's training/validation sides; the split can be reproduced from its manifest. |
| B3: Model comparison | Fit the dummy baseline and compare logistic regression, RBF SVM, and random forest using development data only. | Complete fold results, deterministic selection, fitted winner, and saved preprocessing. |
| B4: Final evaluation | Evaluate the frozen winner and baseline on the test set; generate metrics, predictions, and the model card. | Honest benchmark outcome, uncertainty interval, confusion matrix, and audit checks recorded. |
| B5: Integration and closure | Finish CLI, inference, persistence, documentation, and relevant repository checks. | Saved-model inference reproduces predictions; the engineering and benchmark checklists are completed. |
| V1: Market data — complete | Freeze provider profiles and implement immutable acquisition, local preparation, schema, calendar, and hash checks. | Local SPX snapshot validates; Tiingo fetcher is ready for a user-owned token; focused tests pass. |
| V2: Point-in-time dataset — complete | Build features, forward labels, fold thresholds, and purged yearly splits. | 3,924 SPX origins; eight development folds; window-level and boundary tests prove timing and purge constraints. |
| V3: Model comparison | Implement baselines, the frozen 20-candidate grid, selection, and immutable artifacts. | Development-only selection is reproducible for either profile. |
| V4: Final evaluation | Evaluate the frozen model on the selected profile's untouched holdout. | Metrics, dependence-aware interval, report, and research gate are recorded. |

## Banknote completion gate

- [x] Real UCI data were acquired and their provenance and license were recorded.
- [x] The loader, duplicate policy, frozen split, and development-only preprocessing are implemented.
- [x] All three model families and the majority baseline were evaluated according to the spec.
- [x] The chosen model, full configuration, environment, predictions, and report are reproducible.
- [x] The test accuracy target is measured against the actual deduplicated test population.
- [x] The report separately records engineering completion and benchmark target attainment.
- [x] Focused tests and repository validation pass, or pre-existing failures are identified.
- [x] The final research writeup links to evidence, limitations, and the volatility follow-on.

Target attainment requires all numerical gates in the full spec. If those gates are missed, finish
the implementation, retain the failed result, and document the outcome before starting volatility.
Do not repeatedly change seeds, split membership, or the search space against the same exposed
test labels until a passing score appears. A completed experiment with a negative result still
closes the banknote engineering milestone; it does not become a successful 90% benchmark.

## Scope boundaries

The first implementation is a local Python workflow with a Markdown report and machine-readable
results. It needs no GPU, dashboard, deployed API, experiment-tracking service, or broker connection.
Actual banknote photos, live counterfeit screening, trading signals, and portfolio integration are
outside the banknote scope.

Use the existing `research` dependency extra for scikit-learn. Keep reusable logic under
`src/qr_haven`, research writeups under `docs/research`, configuration under `configs`, and generated
files under the repository's ignored data/artifact directories. Preserve the existing NumPy LSTM
and base-package import behavior.

## Verification during implementation

Use deterministic small fixtures for normal tests; downloading UCI data is an explicit integration
step. After focused checks pass, run the repository's required `pytest`, `ruff check .`, and
`mypy src` checks. Capture the real-data benchmark independently of ordinary CI so a network issue
or an unmet research target cannot be disguised as a parser or software test result.

## Current evidence

Completed 2026-10-06 in America/New_York. The official archive matches the supplied local file:
1,372 raw rows, 24 repeated rows removed across 11 groups, 1,348 retained samples, 1,078 development
and 270 test rows. The reviewed source hash and attribution are in
[`configs/datasets/banknote_authentication.json`](../../configs/datasets/banknote_authentication.json).
The authoritative numeric-to-semantic mapping remains unverified, so outputs use class_0/class_1.

Run `banknote-v1` compared 29 learned candidates plus the majority baseline in five development
folds. RBF SVM (`C=10`, `gamma=scale`) won and achieved **270/270 test predictions correct**.
All numerical/audit gates passed. The majority baseline scored 54.81%; the Wilson accuracy
interval was 98.60%–100.00%. Test outcomes were not used to change the protocol.

Engineering verification: 807 repository tests passed, including 51 new tests. Focused Ruff/mypy
checks pass. Repository-wide Ruff still has 184 pre-existing findings; mypy has 65 errors in 17
existing files and none in the new code. Saved-model inference and immutable repeat evaluation
were verified. The [research writeup](../../docs/research/classification/banknote_authentication.md)
links to all result and verification evidence. Volatility V1 now has a frozen dual-source contract;
the supplied SPX file validates with 3,989 XNYS sessions from 2005-01-03 through 2020-11-04. V2
produces 3,924 eligible origins, 3,207 final-training origins, and 498 sealed holdout origins without
summarizing holdout outcomes.
