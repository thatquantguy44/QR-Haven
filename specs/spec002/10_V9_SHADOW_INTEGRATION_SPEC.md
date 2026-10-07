# Spec002 V9: daily shadow integration

Protocol: `volatility-shadow-v1`. Status: implementation in progress.

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
