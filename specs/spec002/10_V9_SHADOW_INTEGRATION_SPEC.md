# Spec002 V9: daily shadow integration

Protocol: `volatility-shadow-v1`. Status: implemented; container and API acceptance passed.
Rendered browser review remains blocked by the browser runtime. Live scheduling and Power BI
service refresh are not enabled.

Implement the operational workflow described in
[the integration guide](../../docs/research/classification/volatility_shadow_integration.md).
Use the unchanged, hash-verified V8A candidate. The deployment status remains `shadow_only`.

## Contract

- Accept verified Tiingo SPY snapshots with adjusted OHLC and complete XNYS sessions.
- Calculate all 20 frozen features through the origin without requiring future observations.
- Live runs use actual UTC time, require the latest completed session, and finish forecasting
  before the next session opens. Backdated execution is available only in labeled replay mode.
- Persist model identity, inputs, timestamps, and forecasts before scoring matured outcomes.
- Separate immutable forecasts and outcomes in a transactional ledger. Never mix live and replay.
- Mature truth only after five complete sessions, using a single adjusted-price basis.
- Publish versioned reporting generations atomically; leave pending truth and losses null.
- Provide a visual forecast/error/alert/freshness view and typed PBI tables, M queries, and DAX.
- Keep frozen research artifacts unchanged; report operational evidence without promotion claims.

## Required verification

Feature parity with V8A; no future dependence; session-close/next-open checks; five-session maturity;
idempotent retries; immutable records; concurrent-run protection; model identity changes rejected;
pending and unscorable outcomes excluded from losses; report/PBI reconciliation; API prediction
parity and malformed-input handling; actual container build/start and browser inspection where
available. Preserve evidence of failures and any environment blockers.

## Acceptance evidence — 2026-10-07

| Requested work | Result and evidence |
| --- | --- |
| 1. Container/API parity and visual acceptance | Docker image built and the replay API started successfully on `127.0.0.1:8000`; container health is `healthy`. All 184 saved V8A forecasts match HTTP predictions at `rtol=1e-12`, `atol=1e-14`. Invalid requests return 422; dashboard, status, and CSV routes pass. Rendered browser inspection is pending. |
| 2. Daily feature and forecast workflow | Saved snapshots and explicit Tiingo acquisition feed the same causal 20-feature builder. All 184 real-data feature rows match V8A. The calendar-aware live scheduler is implemented, but no live worker is enabled. |
| 3. Forecast ledger and matured outcomes | Saved-data replay produced 189 origins, 184 scored outcomes, and five pending origins. A repeat added zero forecasts and zero outcomes. Tests cover immutability, concurrent retries, stale data, execution timing, five-session maturity, and recovery after a scoring failure. |
| 4. Visual monitoring and Power BI exports | Each run publishes a complete immutable report generation and switches the current pointer atomically. Dashboard data contain 378 model/origin rows. Six typed tables, Power Query scripts, DAX, schema, and a dictionary are generated. Desktop/service import and scheduled refresh have not been executed in Power BI. |

The replay source ends on 2026-10-02. Reporting refreshed on 2026-10-07 correctly marks it stale
against the latest completed 2026-10-07 session. The replay has no gaps within its source window;
this does not establish current live coverage. Five pending outcomes remain null because their
full future windows are unavailable.

Verification results:

- Full repository suite: **909 passed**, with 92 dependency/runtime warnings.
- Focused volatility Ruff checks: no findings. Focused deployment/shadow mypy: five files pass.
- Repository-wide checks retain 184 existing Ruff findings and 65 mypy errors in 17 other files.
- Dashboard JavaScript syntax passes; embedded JSON parses and contains all 378 rows; there are
  no external script tags. These checks do not establish rendered usability.
- Exported CSV values reconcile with embedded dashboard data; report hashes, dimension keys,
  and null pending losses pass independent checks.
- The installed Docker package contains the dashboard template and runs as UID 10001.
- Browser runtime initialization failed with `Importing module "node:process" is not allowed in
  node_repl`. No rendered inspection or screenshot acceptance is claimed.

Saved local evidence under
`artifacts/classification/volatility/tiingo-spy-v1/shadow/replay-v1/` includes
`checks/api-parity.json` (in-process verification), `checks/http-parity.json` (Docker HTTP parity),
`checks/report-reconciliation.json`, the immutable ledger/source history, and the current report
generation. Generated artifacts remain ignored by Git. The integration guide documents commands
to reproduce them.

## Remaining operational steps

1. Review the rendered dashboard and exercise its date/status filters and CSV download once
   browser access is working. Check a pending-only and an empty selection as well as the full view.
2. Configure `TIINGO_API_TOKEN` in the runtime environment and enable the daily worker using a
   separate `live-v1` ledger. Verify the first timely forecast and its first five-session outcome.
3. Import the generated tables and measures in Power BI, validate relationships and totals, then
   configure any required gateway and scheduled refresh.

The model remains `shadow_only` with research status `target_not_met`. Operational acceptance is
separate from promotion. Any future independent promotion test needs a predeclared unseen window;
the full-year 2026 population overlaps already exposed V8A observations.
