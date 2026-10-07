# SPX volatility: four exploratory experiments

Status: implemented and run; exploratory development comparisons only.

Measured: 2026-10-06 America/New_York. Run: `improvements-v1`, based on frozen V3 `spx-vol-v2`.

The five-year rolling variant had the highest mean yearly balanced accuracy at **70.9408%**,
versus **70.8885%** for the original expanding-history classifier. The difference is only
**0.0523 percentage points**. This small development-score gain is not evidence that the new
variant will generalize better. The original failed V4 research gate remains unchanged.

## Protocol and comparison population

The [protocol](../../../specs/spec002/04_EXPLORATORY_EXPERIMENTS.md) was defined before this run,
after the V4 result was already known. It compares 15 variants (including the original control)
and three baselines on 2,008 validation origins across the original eight yearly folds,
2010–2017. Each variant uses identical validation dates and the original fold-specific truth
thresholds. Rolling windows change training membership and weights, not truth. Models are
ranked by mean yearly balanced accuracy, macro F1, and declared variant order.

The primary metric weights years equally. Class support is uneven: 2012 has only two high
outcomes, while 2017 has none. The original single-class convention is preserved: balanced
accuracy in a single-class year equals recall for that class. Consequently, these means should
be read alongside the per-fold supports and pooled descriptive metrics.

## 1. Diagnose the original errors

The saved V4 predictions confirm the cost of the original model's increased sensitivity:

| Saved 2018–2019 prediction | False alarms | Misses | High outcomes detected |
| --- | ---: | ---: | ---: |
| Original classifier | 86 | 41 | 68 / 109 |
| Persistence | 48 | 48 | 61 / 109 |

The paired comparison explains the net difference. The classifier found 20 high outcomes that
persistence missed, but missed 13 that persistence found: seven extra detections overall. It also
introduced 50 false alarms and avoided 12 persistence false alarms: 38 extra false alarms overall.

The year breakdown is informative. In 2018, the classifier had 49 false alarms and 20 misses,
versus persistence's 23 and 28. In 2019, it had 37 false alarms and 21 misses, versus 25 and 20.
Thus the high-recall gain was concentrated in 2018.

The model's 50 extra false alarms occurred where current five-session volatility was below the
threshold: 21 in the `calm` descriptor and 29 in `cooling`. In `cooling`, twenty-session volatility
remained above the threshold. These descriptors describe observed features; they do not establish
why the model made an error. The output includes each date and consecutive FP/FN runs, counted
as overlapping forecast origins rather than independent market events.

## 2–4. Development comparisons

The table shows representative variants; the generated report contains every configuration.
False alarms and misses are totals over all development validation origins. Accuracy and balanced
accuracy are equally weighted yearly means, so they are not computed directly from those totals.

| Experiment / variant | Mean balanced accuracy | Change vs original (pp) | Mean accuracy | False alarms | Misses |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original: inverse-frequency weights, cutoff 0.5 | 70.89% | 0.00 | 87.64% | 182 | 67 |
| Same weights, cutoff 0.6 | 69.62% | -1.27 | 89.82% | 121 | 84 |
| Same weights, cutoff 0.7 | 68.50% | -2.39 | 91.22% | 70 | 107 |
| Half-strength weights, cutoff 0.5 | 68.39% | -2.50 | 89.88% | 106 | 98 |
| Uniform weights, cutoff 0.5 | 68.38% | -2.51 | 91.62% | 59 | 110 |
| EWMA, decay 0.90 | 69.23% | -1.66 | 91.57% | 75 | 95 |
| Linear volatility regression | 67.70% | -3.19 | 91.62% | 57 | 112 |
| Five-year rolling training | 70.94% | +0.05 | 88.34% | 164 | 71 |
| Three-year rolling training | 65.61% | -5.28 | 85.61% | 189 | 101 |
| Persistence reference | 66.29% | -4.60 | 88.69% | 116 | 112 |

**Weights/cutoffs:** all eight changes scored below the original balanced-accuracy control.
Higher cutoffs and weaker class balancing reduced false alarms but sacrificed high-outcome recall.
The plain accuracy increase does not meet the original objective of better balanced accuracy.

**Simple forecasts:** EWMA decay 0.90 led the simple candidates. It beat development persistence
by 2.94 percentage points, but trailed the original classifier by 1.66 points. The regression used
only trailing 5-, 20-, and 60-session volatility to forecast continuous forward volatility, then
applied the same high/normal threshold.

**Rolling windows:** five years narrowly led the daily comparison, with 18 fewer false alarms and
four more misses than the expanding control. The three-year history performed worse. The
every-fifth-origin secondary view also favored five years (79.12% versus 74.84% for the control),
but its smaller sample and different origin selection do not replace the primary daily comparison.

No new holdout metric or confirmatory improvement interval was computed for these variants.
The original control also benefited from selection on these development folds; this exercise is
useful for forming the next experiment, not estimating its unbiased final performance.

## Evidence, verification, and next decision

The complete report and CSVs are local under the Git-ignored directory:

```text
artifacts/classification/volatility/spx-local-v1/experiments/spx-vol-v2/improvements-v1/
```

The [CLI guide](../../api/volatility_market_data.md#four-exploratory-experiments-after-v4)
shows how to verify and open them. Configuration was saved before fitting. The manifest records
input hashes, code hashes, package versions, output hashes, and exploratory status. The original
control reproduced the saved V3 classes and scores (absolute score tolerance `1e-10`). All 17
original V3/V4/ledger files were hash-checked and remained unchanged. No new variant used the
exposed holdout or quarantined 2020 observations; holdout diagnosis read saved V4 predictions.

Verification: **874 repository tests passed**, including eight new experiment tests. These test
causal EWMA/OLS forecasts, exact control weights, purged rolling membership, paired error counts,
CLI execution, identical validation truths, holdout isolation, failure reservation, and artifact
integrity. Focused Ruff and volatility mypy checks pass. Repository-wide Ruff retains 184 existing
findings; mypy retains 65 errors in 17 existing files, with none in the new modules.

There is no compelling primary-metric improvement to promote yet. The five-year window and
EWMA are reasonable candidates for a separately frozen comparison on an untouched future period.
Any such study must define the new source/profile, thresholds, selection rules, and evaluation
period before looking at outcomes. These experiments neither revise the V4 result nor establish
a trading strategy's profitability.
