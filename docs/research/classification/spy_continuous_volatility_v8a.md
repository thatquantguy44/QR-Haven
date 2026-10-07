# SPY continuous volatility V8A: provisional 2026 YTD evaluation

Status: **provisional promotion target not met**.

The V8 candidate was evaluated once on the frozen 2026 YTD population after its model, threshold,
population, uncertainty calculation, and five-condition gate had been committed. The evaluation
used 184 origins from 2026-01-02 through 2026-09-25. Thirteen outcomes exceeded the fixed
high-volatility threshold of 0.197901781191757.

## Forecast results

| Measure | Candidate | Persistence |
| --- | ---: | ---: |
| QLIKE | 0.357035 | 0.811406 |
| Absolute log-volatility error | 0.277507 | 0.390087 |
| Volatility MAE | 0.031609 | 0.045248 |
| Volatility RMSE | 0.043301 | 0.060279 |
| High-volatility recall | 0.076923 | 0.153846 |
| Five-to-one alert cost | 0.336957 | 0.358696 |

The paired QLIKE improvement was +0.454371. Its 95% circular moving-block bootstrap interval was
[+0.132878, +0.878362], based on 2,000 replicates, 20-origin blocks, and seed 5413. The interval is
strictly above zero, supporting a real improvement in continuous forecast loss on this population.

## Promotion decision

| Frozen requirement | Result |
| --- | --- |
| Candidate QLIKE lower than persistence | Passed |
| Candidate log error lower than persistence | Passed |
| QLIKE improvement interval strictly above zero | Passed |
| Candidate high-volatility recall at least persistence | **Failed** |
| Candidate alert cost no greater than persistence | Passed |

The candidate produced 1 true positive, 12 false negatives, 2 false positives, and 169 true
negatives. Persistence produced 2 true positives, 11 false negatives, 11 false positives, and 160
true negatives. The candidate reduced false alarms and overall five-to-one alert cost, but missed
one high-volatility event that persistence detected. Because the protocol requires all five
conditions, V8A reports `target_not_met`.

This is strong evidence for the model as a continuous volatility forecast, but it is not a passed
provisional promotion gate for the fixed alert decision. The 2026 YTD outcomes are now exposed and
cannot support threshold or model retuning. The separately frozen complete-calendar-2026 V8 gate
remains pending.

Generated evidence is stored locally under:

```text
artifacts/classification/volatility/tiingo-spy-v1/continuous_ytd/
├── dataset-v1/
├── candidate-v1/
└── evaluations/candidate-v1/2026-ytd-v1/
```

The evaluation report, metrics, row forecasts and losses, bootstrap samples, exposure ledger, and
hash-bound manifest pass independent verification. The source extension contains 253 verified XNYS
sessions from 2025-10-01 through 2026-10-02, and its overlap with the original 2025 snapshot matched
exactly. See the [frozen protocol](../../../specs/spec002/09_V8A_YTD_PROMOTION_SPEC.md) and
[execution guide](../../api/volatility_market_data.md#v8a-provisional-2026-ytd-gate).
