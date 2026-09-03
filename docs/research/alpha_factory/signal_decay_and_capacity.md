# Signal Decay and Capacity Analysis

## Hypothesis

An alpha's cross-sectional predictive power (IC) decays as the forecast horizon lengthens, and the
rate of that decay — together with market-impact cost growth as NAV scales — jointly determines how
often a strategy must rebalance and how large it can grow before cost drag consumes its edge. Both
questions can be answered directly from artifacts QR Haven already produces (alpha score panels and
`ResearchPipeline` cost attribution) without a new data source or model.

## Literature Review

Signal decay and capacity are standard diagnostics in cross-sectional equity alpha research
(Grinold & Kahn's *Active Portfolio Management* IC/IR framework; the "alpha decay" and "capacity"
sections of most systematic-equity practitioner literature). The core ideas used here:

- **IC decay by horizon** — an alpha's rank IC against forward returns typically falls off with
  horizon length; the horizon at which it halves is a practical proxy for how often the alpha needs
  to be re-estimated and traded.
- **IC information ratio** — `mean(IC) / std(IC)` over time, a standard measure of whether an
  alpha's edge is consistent (high IR) or concentrated in a few lucky periods (low IR), independent
  of its average magnitude.
- **Capacity via cost-drag sweep** — holding the alpha and rebalance schedule fixed and scaling NAV
  up increases each trade's participation rate against ADV, which raises square-root/Almgren-Chriss
  market-impact cost. The NAV at which net Sharpe falls below a fraction of its small-NAV baseline is
  a standard, model-consistent capacity estimate (rather than a rule-of-thumb ADV percentage).

## Data and Point-in-Time Assumptions

- Signal decay diagnostics consume the same wide `returns` (DatetimeIndex x assets) and
  `alpha_scores` (DatetimeIndex x assets) panels already used by `ResearchPipeline`. No new data
  source is introduced.
- Forward returns are constructed to be look-ahead safe: the value at date `t` compounds periods
  `t+1` through `t+horizon`, excluding the return already realized as of `t`. An alpha score dated
  `t` is therefore only ever compared against returns generated strictly after it.
- Capacity sweeps reuse whatever `returns`, `alpha_scores`, and `adv_usd` a caller already has; the
  analysis assumes ADV is held fixed across the NAV grid (a strategy is being asked "how big could I
  run this against today's liquidity", not modeling ADV's own growth).

## Methodology

**Signal decay** (`qr_haven.alpha.decay`):

1. `calculate_forward_returns(returns, horizon)` compounds log returns over the forward window and
   aligns the result back to the signal date.
2. `calculate_ic_by_horizon(alpha_scores, returns, horizons)` computes the mean cross-sectional
   Pearson IC and Spearman rank IC at each horizon, plus the observation count backing each mean.
3. `SignalDecayProfile.half_life()` linearly interpolates the horizon at which rank IC magnitude
   first crosses half of the shortest-horizon value.
4. `calculate_ic_stability(alpha_scores, returns, horizon)` produces the full rank-IC time series at
   one horizon plus its mean, volatility, information ratio, and hit rate.

**Capacity** (`qr_haven.research.capacity`):

1. `sweep_capacity(...)` re-runs `ResearchPipeline` once per NAV in a caller-supplied grid, holding
   the optimizer, borrow schedule, financing rates, impact model, and rebalance cadence fixed via
   `dataclasses.replace` on a single template `PipelineConfig` — only `nav` changes between points.
2. Each `CapacityPoint` records net Sharpe, cost drag, total cost, and total market-impact cost from
   that run's `PipelineResult.summary()` / `CostBreakdown.summary()` — no new cost model is
   introduced, this only re-applies the existing ones at different scale.
3. `CapacityCurve.capacity_nav(sharpe_floor_fraction)` returns the largest swept NAV whose net Sharpe
   still meets a fraction of the smallest-NAV point's Sharpe (the least cost-constrained point in the
   sweep is the baseline).

Borrow and financing costs are expressed by their models as a rate on NAV, so they do not change the
shape of the capacity curve — market impact is the mechanism this analysis isolates.

## Implementation

- `src/qr_haven/alpha/decay.py` — `SignalDecayProfile`, `ICStability`, `calculate_forward_returns`,
  `calculate_ic_by_horizon`, `calculate_rolling_ic`, `calculate_ic_stability`.
- `src/qr_haven/research/capacity.py` — `CapacityPoint`, `CapacityCurve`, `sweep_capacity`.
- Both are exported from their package `__init__.py` (`qr_haven.alpha`, `qr_haven.research`).

## Results

Validated on synthetic data (see Tests): a signal constructed as a perfect one-period-ahead
predictor shows rank IC ≈ 1.0 at horizon 1, decaying toward the theoretical `1/sqrt(horizon)` bound
for i.i.d. returns as horizon grows, with a finite, interpolatable half-life. A capacity sweep with a
fixed square-root impact model and fixed ADV shows market-impact cost and cost drag rising
monotonically with NAV, and net Sharpe degrading in step — reproducing the expected capacity-curve
shape end to end through the existing cost models.

## Limitations

- IC decay is measured on realized correlation, not a fitted decay-curve model; `half_life()` is a
  linear interpolation between the two horizons bracketing the crossing point, not a smoothed fit.
- `calculate_ic_by_horizon` iterates per date in Python; adequate for research-scale panels but not
  optimized for very large universes or very long histories.
- Capacity analysis assumes ADV does not grow with NAV and that the strategy's target weights are
  unchanged by its own cost impact (no feedback from cost into the optimizer's objective).
- `sweep_capacity` re-runs the full walk-forward pipeline once per grid point; for fine-grained NAV
  grids over long histories this is the dominant cost, not the analysis itself.

## Future Work

- Fit a parametric decay curve (e.g. exponential) instead of linear interpolation for `half_life()`.
- Extend `sweep_capacity` to also vary participation-rate assumptions, not just NAV.
- Add market_terminal panels for `SignalDecayProfile` and `CapacityCurve` alongside the existing
  `PipelineResult` panels in `qr_haven.integrations.market_terminal.research`.
- Feed `CapacityCurve.capacity_nav()` back into `qr_haven.alpha` model ranking so multiple candidate
  alphas can be compared on capacity-adjusted, not just raw, Sharpe.

## Source Modules

- `src/qr_haven/alpha/decay.py`
- `src/qr_haven/research/capacity.py`
- `src/qr_haven/alpha/combination.py` (`information_coefficient`, reused for rank IC)
- `src/qr_haven/research/pipeline.py` (`ResearchPipeline`, reused unmodified for capacity sweeps)

## Tests

- `tests/test_alpha_decay.py` — forward-return construction, IC-by-horizon decay shape, half-life,
  rolling IC, and IC stability diagnostics.
- `tests/test_capacity.py` — NAV sweep mechanics, monotonic impact-cost and cost-drag growth,
  `capacity_nav` threshold logic, and `to_frame()` serialization.

## Terminal or Reporting Handoff

Not yet wired into `market_terminal` — see Future Work. `CapacityCurve.to_frame()` already returns a
NAV-indexed `pandas.DataFrame` suitable for an ad hoc terminal table panel using the same
`TerminalPanel` contract as the existing `PipelineResult` panels.
