# SPY adaptive-target training-history experiment

Status: development selection complete; no replacement final model selected.

Measured: 2026-10-07 America/New_York.

## Question and evidence boundary

V7 asked whether the histogram classifier benefits from more history after replacing the fixed
high-volatility threshold with a causal regime-adaptive threshold. The protocol was committed
before implementation and comparison. The workflow read only the verified Tiingo SPY development
observations through 2023. It did not open the 2024–2025 outcomes or load their evaluation
artifacts.

At each origin, high volatility means forward five-session volatility strictly above the 75th
percentile of outcomes observable during the preceding three calendar years. Each threshold uses
at least 500 outcomes whose label end precedes the origin. This creates one shared point-in-time
truth series for every candidate.

The comparison used the existing histogram classifier and 20 challenger features. It tested
five-, eight-, and twelve-year rolling histories, expanding history, and expanding history with a
five-year exponential half-life. Each design was combined with adaptive persistence at five fixed
weights and calibrated from the preceding three out-of-fold years.

## Development result

The experiment produced 4,211 eligible rows. Selection used 2,763 origins in calendar-year folds
from 2013 through 2023. Pure adaptive persistence ranked first:

| Candidate | Model weight | Mean yearly balanced accuracy | Mean macro F1 | Mean high recall |
| --- | ---: | ---: | ---: | ---: |
| Adaptive persistence | 0.00 | 0.716707 | 0.625748 | 0.409873 |
| Expanding blend | 0.75 | 0.713764 | 0.615274 | 0.342042 |
| Five-year blend | 0.75 | 0.713594 | 0.607558 | 0.348922 |
| Twelve-year model only | 1.00 | 0.694039 | 0.593906 | 0.332322 |
| Five-year model only | 1.00 | 0.686054 | 0.594290 | 0.345556 |
| Eight-year model only | 1.00 | 0.671100 | 0.584277 | 0.330936 |
| Decayed expanding model only | 1.00 | 0.669368 | 0.572379 | 0.316989 |

The selected artifact names `rolling_5y` because ranking requires a deterministic history-design
tie break. Its model weight is zero, so training history does not affect its predictions. Model
weights 0.25 and 0.50 also produced the same hard classifications as persistence at the fixed 0.5
cutoff.

Using twelve years improved the standalone model by 0.80 balanced-accuracy points relative to five
years. Expanding history slightly improved the 75% model blend by about 0.02 points. Neither gain
was enough to beat adaptive persistence. The evidence therefore supports the narrow conclusion
that additional history can help this fitted model, while rejecting the stronger claim that it
produces a better classifier under the current feature, calibration, and decision rules.

## Adaptive-target behavior

Across the outer folds, 693 of 2,763 origins were high volatility, or 25.08%. This is close to the
intended 25% rate in aggregate and avoids the 12.3% high-state prevalence seen in the fixed-threshold
2024–2025 evaluation.

The target remained unstable by calendar year. High-state prevalence ranged from zero in 2017 and
2023 to 48.62% in 2020 and 53.78% in 2022. A trailing quantile adapts after a regime changes; it
does not guarantee 25% positives within each future year. Two single-class validation years also
make mean yearly balanced accuracy less informative than it would be with both classes in every
fold. The pooled persistence balanced accuracy was 0.733722, with 0.598846 high recall, but pooled
metrics do not replace the predeclared yearly selection rule.

## Outputs and conclusion

Generated evidence is stored locally under:

```text
artifacts/classification/volatility/tiingo-spy-v1/history_experiments/v7-history-v1/
```

The run saves each causal threshold and audit boundary, exact training membership and weights,
calibration rows, individual predictions, fold and pooled metrics, target prevalence, ranking,
selection, report, and a hash-bound manifest. Verification confirms that evaluation outcomes were
not opened and no final model was fitted. The full repository suite passes 884 tests; focused Ruff
and strict mypy checks pass for the V7 implementation.

V7 does not justify testing another fitted classifier on new data. A more useful next research
direction is continuous volatility forecasting evaluated with forecast loss, followed by a
decision threshold chosen for a stated risk-management cost. That separates forecast quality from
the unstable binary label and can still compare against persistence without reopening 2024–2025
as fresh evidence.

See the [frozen protocol](../../../specs/spec002/07_V7_HISTORY_ADAPTATION_SPEC.md) and the
[execution guide](../../api/volatility_market_data.md#v7-adaptive-target-history-experiment).
