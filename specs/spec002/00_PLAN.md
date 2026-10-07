# Spec002 Plan: Supervised Finance Classification

Status: banknote milestones B1–B5 complete; benchmark passed. Volatility V1–V7 are complete. V8
development selected a continuous-volatility candidate for a prospective 2026 promotion test. The
local SPX and independent Tiingo SPY final research gates remain unmet.

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
| [04_EXPLORATORY_EXPERIMENTS.md](04_EXPLORATORY_EXPERIMENTS.md) | Saved-error diagnosis, weights/cutoffs, simple forecasts, and rolling windows after V4. |
| [05_FROZEN_CHALLENGER_SPEC.md](05_FROZEN_CHALLENGER_SPEC.md) | Frozen five-year challenger with new features, nested calibration/selection, ensemble, and one 2020 evaluation. |
| [06_TIINGO_SPY_CHALLENGER_SPEC.md](06_TIINGO_SPY_CHALLENGER_SPEC.md) | Independent adjusted-SPY challenger with 2013–2023 selection and untouched 2024–2025 evaluation. |
| [07_V7_HISTORY_ADAPTATION_SPEC.md](07_V7_HISTORY_ADAPTATION_SPEC.md) | Development-only adaptive target and training-history comparison after the Tiingo evaluation. |
| [08_V8_CONTINUOUS_VOLATILITY_SPEC.md](08_V8_CONTINUOUS_VOLATILITY_SPEC.md) | Continuous volatility forecasts, decision-cost diagnostics, and the prospective 2026 promotion gate. |
| [09_V8A_YTD_PROMOTION_SPEC.md](09_V8A_YTD_PROMOTION_SPEC.md) | Frozen 2026 YTD provisional gate for the selected continuous-volatility candidate. |

The banknote commands, APIs, configuration and outputs are implemented. See the
[measured research writeup](../../docs/research/classification/banknote_authentication.md) and
[API guide](../../docs/api/classification.md). The volatility workflow now supports validated
SPX/Tiingo profiles, point-in-time features, purged splits, development-only model selection, and
single-use holdout evaluation. The frozen Tiingo SPY replication is complete.

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
| V3: Model comparison — complete for local SPX | Implement baselines, the frozen 20-candidate grid, selection, and immutable artifacts. | Development-only selection is reproducible and the SPX winner is frozen. |
| V4: Final evaluation — complete for local SPX | Evaluate the frozen model on the selected profile's untouched holdout. | Metrics, dependence-aware interval, report, and failed research gate are recorded without retuning. |
| Exploratory follow-up — complete | Implement four experiments without modifying V3/V4 or making new holdout predictions. | `improvements-v1` replays V3 and compares 15 variants plus baselines; immutable reports and diagnostics are saved. |
| V5 challenger — complete; gate not met | Build the five-year HAR-RV/EWMA/histogram comparison, chronological calibration, persistence ensemble, and one-time 2020 test. | Histogram plus persistence was selected; 2020 balanced accuracy beat persistence, but recall and the bootstrap gate failed. |
| V6 Tiingo SPY challenger — complete; gate not met | Apply the fixed challenger grid to adjusted SPY and evaluate once on 2024–2025. | The adjusted snapshot passed calendar validation; the challenger beat persistence by 0.85 balanced-accuracy points, but its bootstrap interval crossed zero. |
| V7 history adaptation — complete | Compare longer and recency-weighted histories under a point-in-time adaptive volatility target, using only data through 2023. | Twelve years modestly improved the model-only score, but pure adaptive persistence won the overall development ranking. |
| V8 continuous forecasting — development complete; promotion pending | Select a continuous five-session volatility forecast through 2023 and reserve a one-time 2026 promotion test. | A 75% histogram / 25% persistence variance blend won development; complete 2026 data are still required for the frozen gate. |
| V8A provisional promotion — protocol frozen | Test the unchanged V8 winner once on untouched origins through 2026-09-25. | Population, fixed alert threshold, bootstrap, and five gates are committed before acquiring the 2026 extension. |

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
summarizing holdout outcomes. V3 compared 20 learned candidates and three baselines across eight
purged yearly folds. Histogram gradient boosting won with mean yearly balanced accuracy 0.708885,
versus 0.662901 for persistence. On the now-exposed 2018–2019 holdout, it scored 0.701387 balanced
accuracy versus 0.718120 for persistence. The bootstrap interval crossed zero, so the research gate
was not met and no post-holdout retuning was performed.

A separate exploratory workflow subsequently compared weights/cutoffs, EWMA/linear forecasts,
and rolling windows on the original development folds, with saved-error diagnosis as the fourth
experiment. The five-year window scored 0.709408 mean yearly balanced accuracy versus 0.708885
for the original control, a small exploratory gain. No replacement model was fitted for holdout
use and the original V4 evidence is unchanged. See the
[experiment results](../../docs/research/classification/spx_volatility_experiments.md).
All 874 repository tests pass; the existing repository-wide lint/type findings are unchanged.

The frozen V5 challenger added downside, range, and realized-volatility features; nested
chronological Platt calibration; HAR-RV and EWMA candidates; and persistence ensembles. Selection
chose the histogram model with 75% model / 25% persistence weight. On the one-time 209-origin 2020
evaluation it scored 0.770085 balanced accuracy versus 0.737019 for persistence, but high-volatility
recall fell to 0.555556 versus 0.812500 and the improvement interval crossed zero. The research gate
was not met. See the
[challenger result](../../docs/research/classification/spx_volatility_challenger.md).
Repository verification now passes 878 tests. Focused challenger lint and type checks pass; the
184 existing repository-wide Ruff findings and 65 mypy errors in 17 other files remain unchanged.

The V6 protocol was frozen before Tiingo acquisition. The immutable adjusted-SPY snapshot contains
5,283 verified XNYS sessions from 2005-01-03 through 2025-12-31. Development selection again chose
the histogram model with 75% model / 25% persistence weight. On the single 497-origin 2024–2025
evaluation, balanced accuracy was 0.662825 versus 0.654290 for persistence, and high-volatility
recall was 0.442623 versus 0.393443. The paired improvement was +0.008535 with a 95% block-bootstrap
interval of [-0.0627, +0.0895], so the research gate was not met. See the
[Tiingo SPY result](../../docs/research/classification/spy_volatility_challenger.md).

V7 then compared five-, eight-, and twelve-year rolling histories, expanding history, and a
five-year-half-life expanding fit under a causal three-year adaptive threshold. It used 4,211
eligible development rows and did not load 2024–2025 outcomes. Pure adaptive persistence ranked
first at 0.716707 mean yearly balanced accuracy. The twelve-year model-only candidate improved on
the five-year model-only candidate, 0.694039 versus 0.686054, but remained below persistence. The
adaptive target produced 25.08% high outcomes overall across the 2013–2023 folds, while individual
years ranged from zero to 53.78%. See the
[V7 result](../../docs/research/classification/spy_volatility_history_v7.md).

V8 directly forecast five-session annualized volatility and ranked 20 model/persistence variance
blends plus persistence using mean yearly QLIKE. The 75% histogram regression / 25% persistence
blend ranked first at 0.510667 QLIKE, versus 1.132493 for persistence, and improved absolute
log-volatility error in all 11 outer years. Its pooled adaptive-alert recall was 0.620491 versus
0.598846, and five-to-one alert cost was 0.539631 versus 0.601520. The candidate advances to the
predeclared 2026 evaluation, but has not passed that promotion gate. See the
[V8 development result](../../docs/research/classification/spy_continuous_volatility_v8.md).
