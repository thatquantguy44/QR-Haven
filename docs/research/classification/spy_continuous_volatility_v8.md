# SPY continuous volatility V8: development selection

Status: candidate selected; prospective 2026 promotion evaluation pending.

Measured: 2026-10-07 America/New_York.

## Objective and evidence boundary

V8 predicts annualized realized volatility over the next five sessions directly. This separates
forecast quality from the unstable binary high-volatility label observed in V6 and V7. The
protocol and promotion gates were committed before implementation or comparison.

Development used only verified Tiingo SPY observations through 2023. The workflow did not open
2024–2025 outcomes or load their evaluation artifacts. It did not fit a final model. Calendar 2026
remains reserved for the one-time promotion test after its complete data extension is available.

## Candidates and selection

The comparison included histogram gradient boosting regression, HAR-RV, and EWMA decays 0.90,
0.94, and 0.97. Each base forecast was blended in variance space with persistence at model weights
0.25, 0.50, 0.75, and 1.00. Pure persistence was the twenty-first candidate. Fitted models used
purged twelve-calendar-year histories.

Selection minimized unrounded mean yearly QLIKE across 2013–2023, with absolute log-volatility
error as the first tie break. The winner was
`hist_gradient_boosting_regression_w075`: 75% histogram forecast variance and 25% persistence
variance.

| Candidate | Mean yearly QLIKE | Improvement vs persistence | Log MAE | Volatility MAE | Alert recall | Alert cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 75% histogram / 25% persistence | 0.510667 | +0.621826 | 0.354457 | 0.044038 | 0.407578 | 0.538141 |
| 75% EWMA 0.97 / 25% persistence | 0.526516 | +0.605977 | 0.420446 | 0.054881 | 0.437372 | 0.523370 |
| 50% histogram / 50% persistence | 0.540812 | +0.591682 | 0.368158 | 0.046348 | 0.421620 | 0.534518 |
| Histogram only | 0.553766 | +0.578728 | 0.354045 | 0.043692 | 0.385723 | 0.563848 |
| Persistence | 1.132493 | 0.000000 | 0.452392 | 0.055778 | 0.409873 | 0.599900 |

The selected candidate had lower QLIKE and lower absolute log-volatility error than persistence in
all 11 outer years. The smallest yearly QLIKE improvement was 0.085279 in 2022; the largest was
1.036687 in 2017. This consistency is stronger development evidence than the previous binary
classifiers produced.

## Pooled forecast and alert behavior

Across 2,763 development origins, pooled results were:

| Measure | Selected candidate | Persistence |
| --- | ---: | ---: |
| QLIKE | 0.511533 | 1.134112 |
| Absolute log-volatility error | 0.354641 | 0.452628 |
| Volatility MAE | 0.044083 | 0.055834 |
| Volatility RMSE | 0.071057 | 0.082944 |
| Adaptive-alert balanced accuracy | 0.767733 | 0.733722 |
| Adaptive-alert high recall | 0.620491 | 0.598846 |
| Five-to-one alert cost | 0.539631 | 0.601520 |

The selected forecast detected 430 of 693 adaptive high-volatility outcomes, compared with 415 for
persistence. It produced 176 false alarms and 263 misses, versus 272 false alarms and 278 misses for
persistence. Continuous selection therefore improved forecast loss while also improving the pooled
alert tradeoff. Mean yearly alert recall was slightly lower because 2014 and 2021 were weaker and
2017 and 2023 contained no high outcomes.

## Promotion status

The candidate has passed **development selection**, not the research promotion gate. Promotion
requires a single untouched 2026 evaluation to satisfy all five frozen requirements:

1. lower QLIKE than persistence;
2. lower absolute log-volatility error;
3. a 95% block-bootstrap interval strictly above zero for QLIKE improvement;
4. high-volatility recall at least persistence;
5. five-to-one alert cost no greater than persistence.

The complete 2026 extension is not yet available as of October 7, 2026. Fitting or evaluating on a
partial year would change the frozen population and weaken the intended confirmation. The model can
run in shadow mode, but it cannot be represented as research-approved until the complete gate is
evaluated.

Generated evidence is stored locally under:

```text
artifacts/classification/volatility/tiingo-spy-v1/continuous_experiments/v8-continuous-v1/
```

It contains every forecast and row loss, exact training membership, fit audits, adaptive
thresholds, fold and pooled metrics, ranking, selection, report, and a hash-bound manifest. The full
repository suite passes 890 tests; focused Ruff and strict mypy checks pass for the V8 modules. See the
[frozen protocol](../../../specs/spec002/08_V8_CONTINUOUS_VOLATILITY_SPEC.md) and
[execution guide](../../api/volatility_market_data.md#v8-continuous-volatility-experiment).
