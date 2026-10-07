# Market Regime Detection

Research home for Hidden Markov Models, Bayesian switching, clustering, change point detection, and
regime-aware allocation.

## Queued supervised classification project

The [UCI Banknote Authentication build](../classification/banknote_authentication.md) is complete
and passed its frozen benchmark. The next project is
[next-period high-volatility versus normal-volatility classification](../../../specs/spec002/02_VOLATILITY_FOLLOW_ON.md).

The initial task predicts volatility over the next five trading sessions from information available
at the current close. It uses real historical market data, purged chronological validation, and
majority/persistence baselines. Its objective is useful risk forecasting with honest out-of-sample
results; 90% accuracy is not a promised requirement.

The [implementation plan](../../../specs/spec002/00_PLAN.md) defines the banknote completion gate
and the order of work. The follow-on remains queued and has not been implemented.
