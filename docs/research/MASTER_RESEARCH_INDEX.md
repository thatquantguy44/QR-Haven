# QR Haven Master Research Index

This file is the central map for QR Haven research. Use it to find the current
research areas, implemented modules, API references, and core vocabulary.

`docs/research/` is the canonical home for research-facing documentation.
Source code remains in `src/qr_haven`, and API references remain in `docs/api`.

Last updated: 2026-08-07

## Source Documents

- [Repository vision](../../README.md)
- [Research project template](research_template.md)
- [Repository structure](../architecture/repository_structure.md)
- [Development workflow](../operations/development.md)

## Research Documentation Tree

| Folder | Purpose |
| --- | --- |
| [optimization](optimization/README.md) | Institutional portfolio optimization research and implementation notes. |
| [optimization/Institutional-Portfolio-Optimizer](optimization/Institutional-Portfolio-Optimizer/README.md) | Detailed optimizer, data, risk, backtesting, attribution, and terminal integration docs. |
| [Transaction-Costs](Transaction-Costs/README.md) | Market impact, borrow cost, financing cost, slippage, and attribution research. |
| [Securities-Lending](Securities-Lending/README.md) | Lending revenue, short interest, utilization, and securities-lending alpha features. |
| [alpha_factory](alpha_factory/README.md) | Signal discovery, validation, model research, turnover, capacity, and alpha diagnostics. |
| [market_regime_detection](market_regime_detection/README.md) | HMM, clustering, switching, change point, and regime-aware allocation research. |
| [risk_engine](risk_engine/README.md) | VaR, CVaR, expected shortfall, stress, factor, liquidity, and Monte Carlo research. |
| [execution_simulator](execution_simulator/README.md) | VWAP, TWAP, POV, implementation shortfall, realized spread, and execution simulation. |
| [options_research](options_research/README.md) | Volatility surfaces, SABR, Heston, Greeks, variance risk premium, and dispersion. |
| [fixed_income_research](fixed_income_research/README.md) | Nelson-Siegel, Svensson, carry and roll, duration, convexity, and bond optimization. |
| [ai_research_platform](ai_research_platform/README.md) | Literature, hypothesis, feature discovery, optimization, risk, execution, attribution, and documentation agents. |

## Research Area Map

| Area | Research home | Primary docs | Source modules | Status |
| --- | --- | --- | --- | --- |
| Institutional portfolio optimizer | [optimization](optimization/README.md) | [optimizer docs](optimization/Institutional-Portfolio-Optimizer/README.md), [returns and optimization](optimization/Institutional-Portfolio-Optimizer/returns_and_optimization.md), [backtesting](optimization/Institutional-Portfolio-Optimizer/backtesting.md), [risk engine](optimization/Institutional-Portfolio-Optimizer/risk_engine.md) | `src/qr_haven/portfolio`, `src/qr_haven/backtesting`, `src/qr_haven/risk`, `src/qr_haven/reporting` | Implemented v0/v1 core |
| Transaction cost models | [Transaction-Costs](Transaction-Costs/README.md) | [market impact](Transaction-Costs/market_impact.md), [borrow cost](Transaction-Costs/borrow_cost.md), [financing cost](Transaction-Costs/financing_cost.md), [securities finance attribution](Transaction-Costs/securities_finance_attribution.md) | `src/qr_haven/costs` | Implemented core models |
| Securities lending | [Securities-Lending](Securities-Lending/README.md) | [lending revenue](Securities-Lending/lending_revenue.md), [short interest features](Securities-Lending/short_interest_features.md), [lending revenue API](../api/lending_revenue.md), [short interest API](../api/short_interest_features.md) | `src/qr_haven/costs/lending.py`, `src/qr_haven/features/securities_lending.py`, `src/qr_haven/alpha/borrow_signal.py` | Implemented core features |
| Alpha research factory | [alpha_factory](alpha_factory/README.md) | [feature store API](../api/feature_store.md), [short interest API](../api/short_interest_features.md) | `src/qr_haven/alpha`, `src/qr_haven/features` | Implemented initial factor and signal layer |
| Market regime detection | [market_regime_detection](market_regime_detection/README.md) | Research stub only | `src/qr_haven/regimes` | Implemented GMM/HMM models; docs needed |
| Research pipeline | See repository vision | [terminal integration](optimization/Institutional-Portfolio-Optimizer/terminal_integration.md), [market terminal API](../api/market_terminal_integration.md) | `src/qr_haven/research/pipeline.py`, `src/qr_haven/pipeline.py` | Implemented cost-aware pipeline; reporting expansion recommended |
| Risk engine | [risk_engine](risk_engine/README.md) | [risk engine docs](optimization/Institutional-Portfolio-Optimizer/risk_engine.md), [risk engine API](../api/risk_engine.md) | `src/qr_haven/risk` | Implemented simple portfolio risk engine |
| Execution simulator | [execution_simulator](execution_simulator/README.md) | Research stub only | `src/qr_haven/execution` | Stub |
| Options research | [options_research](options_research/README.md) | Research stub only | `src/qr_haven/options` | Stub |
| Fixed income research | [fixed_income_research](fixed_income_research/README.md) | Research stub only | `src/qr_haven/fixed_income` | Stub |
| AI quant research platform | [ai_research_platform](ai_research_platform/README.md) | Research stub only | `src/qr_haven/ai_agents` | Stub |
| Machine learning methods | See repository vision | None yet | `src/qr_haven/ml` | Stub |

## Implemented Capability Index

### Data and Feature Infrastructure

- CSV and SQLite data ingestion: `src/qr_haven/data`
- Return matrix construction: `src/qr_haven/data/returns.py`
- Feature store and factor helpers: `src/qr_haven/features`
- Securities lending features: `src/qr_haven/features/securities_lending.py`

Reference docs:

- [Data ingestion API](../api/data_ingestion.md)
- [Feature store API](../api/feature_store.md)
- [Short interest features API](../api/short_interest_features.md)
- [Optimizer data ingestion](optimization/Institutional-Portfolio-Optimizer/data_ingestion.md)
- [Securities lending features](Securities-Lending/short_interest_features.md)

### Alpha and Signal Models

- Composite alpha model: `src/qr_haven/alpha/models.py`
- Rank alpha model: `src/qr_haven/alpha/models.py`
- Borrow pressure signal: `src/qr_haven/alpha/borrow_signal.py`
- Cross-sectional score combination and IC utilities: `src/qr_haven/alpha/combination.py`

Open research path:

- Signal decay by holding period
- IC and rank IC stability
- Capacity-aware alpha ranking
- Regression-based alpha selection

### Portfolio Construction and Optimization

- Equal weight optimizer: `src/qr_haven/portfolio/optimizers.py`
- Mean-variance optimizer: `src/qr_haven/portfolio/optimizers.py`
- Constraint diagnostics: `src/qr_haven/portfolio/optimizers.py`

Reference docs:

- [Returns and optimization](optimization/Institutional-Portfolio-Optimizer/returns_and_optimization.md)
- [Optimizer diagnostics](optimization/Institutional-Portfolio-Optimizer/optimizer_diagnostics.md)
- [Returns and optimization API](../api/returns_and_optimization.md)

Open research path:

- Risk parity
- Hierarchical risk parity
- Black-Litterman
- Robust optimization
- Multi-period optimization

### Backtesting, Reporting, and Terminal Handoff

- Walk-forward backtester: `src/qr_haven/backtesting/engine.py`
- Performance attribution: `src/qr_haven/backtesting/attribution.py`
- Portfolio analytics: `src/qr_haven/reporting/analytics.py`
- Backtest report bundle: `src/qr_haven/reporting/backtest_report.py`
- Market terminal panels: `src/qr_haven/integrations/market_terminal`

Reference docs:

- [Backtesting](optimization/Institutional-Portfolio-Optimizer/backtesting.md)
- [Backtesting roadmap](optimization/Institutional-Portfolio-Optimizer/backtesting_roadmap.md)
- [Performance attribution](optimization/Institutional-Portfolio-Optimizer/performance_attribution.md)
- [Terminal integration](optimization/Institutional-Portfolio-Optimizer/terminal_integration.md)
- [Market terminal integration API](../api/market_terminal_integration.md)

Recommended next build:

- Add reporting and terminal panels for `ResearchPipeline` and `PipelineResult`, including net vs.
  gross returns, cost attribution, alpha scores used, and cost drag.

### Cost and Securities Finance Models

- Square-root market impact model: `src/qr_haven/costs/market_impact.py`
- Almgren-Chriss-style market impact model: `src/qr_haven/costs/market_impact.py`
- Borrow cost model: `src/qr_haven/costs/borrow.py`
- Financing cost model: `src/qr_haven/costs/financing.py`
- Lending revenue model: `src/qr_haven/costs/lending.py`
- Short squeeze risk model: `src/qr_haven/costs/squeeze.py`
- Securities finance attribution: `src/qr_haven/costs/securities_finance.py`

Reference docs:

- [Market impact](Transaction-Costs/market_impact.md)
- [Borrow cost](Transaction-Costs/borrow_cost.md)
- [Financing cost](Transaction-Costs/financing_cost.md)
- [Securities finance attribution](Transaction-Costs/securities_finance_attribution.md)
- [Lending revenue](Securities-Lending/lending_revenue.md)

### Regime Detection

- Gaussian mixture model regime detector: `src/qr_haven/regimes/gmm.py`
- Hidden Markov model regime detector: `src/qr_haven/regimes/hmm.py`
- Shared math utilities: `src/qr_haven/regimes/_math.py`

Open research path:

- Regime-conditioned allocation
- Regime-conditioned risk scaling
- Change point detection
- Regime stability diagnostics

## Research Status Board

| Priority | Build or research | Work item | Why it matters |
| --- | --- | --- | --- |
| 1 | Build | `PipelineResult` reporting and terminal panels | Makes the cost-aware research pipeline visible, inspectable, and demo-ready. |
| 2 | Research | Signal decay and capacity analysis | Connects alpha quality, turnover, market impact, borrow costs, and holding-period choice. |
| 3 | Build | CI workflow for pytest, ruff, and mypy | Turns the repository's quality standard into an automatic gate. |
| 4 | Research | Regime-conditioned portfolio constraints | Uses existing GMM/HMM code to change risk budgets and exposure limits by market state. |
| 5 | Build | Execution simulator v0 | Extends market impact into VWAP, TWAP, POV, and implementation shortfall workflows. |
| 6 | Research | Risk parity and HRP | Adds a recognized institutional portfolio construction baseline. |
| 7 | Research | Options volatility surface v0 | Opens the options research track with Greeks and volatility surface primitives. |
| 8 | Research | Fixed income curve model v0 | Opens the fixed income track with Nelson-Siegel/Svensson curve fitting. |
| 9 | Build | AI research memo generator | Converts backtest and pipeline outputs into structured research memos. |

## Quant Dictionary

| Term | Definition | QR Haven location |
| --- | --- | --- |
| Alpha | A forecast signal expected to explain or predict future asset returns. | `src/qr_haven/alpha` |
| Alpha score | Cross-sectional value assigned to an asset by an alpha model. Higher usually means more attractive. | `src/qr_haven/alpha/models.py` |
| ADV | Average daily dollar volume; used to estimate trading participation and market impact. | `src/qr_haven/research/pipeline.py` |
| Almgren-Chriss | Execution cost framework separating temporary and permanent impact. | `src/qr_haven/costs/market_impact.py` |
| Backtest | Historical simulation of a strategy using rules known at each point in time. | `src/qr_haven/backtesting/engine.py` |
| Borrow cost | Fee paid to borrow securities for short positions. | `src/qr_haven/costs/borrow.py` |
| Capacity | Strategy size at which turnover, liquidity, borrow, or impact costs materially degrade returns. | Research topic |
| Constraint pressure | Diagnostic showing how close optimized weights are to portfolio limits. | `src/qr_haven/portfolio/optimizers.py` |
| Cost drag | Difference between gross strategy returns and net-of-cost returns. | `src/qr_haven/research/pipeline.py` |
| Covariance matrix | Matrix of asset return variance and co-movement used by optimizers and risk models. | `src/qr_haven/portfolio/optimizers.py` |
| Drawdown | Decline from a prior equity-curve peak. | `src/qr_haven/reporting/analytics.py` |
| Expected return | Forecast or historical estimate of average asset return used by an optimizer. | `src/qr_haven/portfolio/optimizers.py` |
| Financing cost | Interest cost or credit from leverage, debit balances, and short-sale proceeds. | `src/qr_haven/costs/financing.py` |
| Gross exposure | Sum of absolute portfolio weights. A 130/30 portfolio has 160 percent gross exposure. | `src/qr_haven/portfolio/optimizers.py` |
| HMM | Hidden Markov model used to infer latent market regimes. | `src/qr_haven/regimes/hmm.py` |
| IC | Information coefficient; correlation between signal scores and forward returns. | `src/qr_haven/alpha/combination.py` |
| Long-only | Portfolio constraint disallowing negative weights. | `src/qr_haven/portfolio/optimizers.py` |
| Market impact | Trading cost caused by order size relative to liquidity and volatility. | `src/qr_haven/costs/market_impact.py` |
| Mean variance | Optimization framework balancing expected return against portfolio variance. | `src/qr_haven/portfolio/optimizers.py` |
| NAV | Net asset value; portfolio dollar capital base used to convert weights to dollars. | `src/qr_haven/research/pipeline.py` |
| Net exposure | Sum of signed portfolio weights. | `src/qr_haven/portfolio/optimizers.py` |
| PipelineResult | Cost-aware end-to-end research output containing net returns, gross returns, weights, costs, alpha scores, turnover, and risk metrics. | `src/qr_haven/research/pipeline.py` |
| Point-in-time | Data discipline requiring that only information available as of the evaluation date is used. | Research standard |
| Rank IC | Spearman rank correlation between signal ranks and forward return ranks. | `src/qr_haven/alpha/combination.py` |
| Rebalance | Date on which target weights are recomputed and trades are applied. | `src/qr_haven/backtesting/engine.py` |
| Regime | Market state such as low volatility, crisis, bull, or bear environment. | `src/qr_haven/regimes` |
| Sharpe ratio | Annualized excess return divided by annualized volatility. | `src/qr_haven/risk/engine.py` |
| Short interest | Measure of borrowed or shorted shares, often used for crowding and squeeze risk signals. | `src/qr_haven/features/securities_lending.py` |
| Short squeeze | Risk that crowded short positions rally sharply as shorts cover. | `src/qr_haven/costs/squeeze.py` |
| Turnover | Sum of absolute weight changes at a rebalance. | `src/qr_haven/backtesting/engine.py` |
| VaR | Value at Risk; estimated loss threshold over a given confidence level and horizon. | `src/qr_haven/risk/engine.py` |
| Walk-forward validation | Repeated train/estimate then forward-test process over rolling time windows. | `src/qr_haven/backtesting/engine.py` |

## Research Writeup Checklist

Every promoted research project should include:

- Hypothesis
- Literature review
- Data and point-in-time assumptions
- Methodology
- Implementation links
- Results
- Limitations
- Future work
- Source modules
- Tests
- Terminal or reporting handoff, when relevant

Use [research_template.md](research_template.md) for the canonical writeup shape.

## Documentation Gaps

- Regime detection has implemented source and tests but needs research/API docs.
- `ResearchPipeline` has strong implementation coverage but needs dedicated docs and terminal panels.
- Execution, options, fixed income, ML, and AI agents are package stubs and need first scoped projects.
- CI is expected by the development docs but the workflow is not yet implemented.
- Several research area READMEs in `docs/research/` are still short placeholders and should be
  expanded as projects mature.
