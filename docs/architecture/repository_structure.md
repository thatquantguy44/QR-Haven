# Repository Structure

QR-Haven is organized as one integrated research platform. Shared infrastructure lives in
`src/qr_haven`, while project-specific research artifacts live under `research/`, `notebooks/`,
`docs/`, and `dashboards/`.

```text
QR-Haven/
├── configs/                  # Hydra/YAML-style runtime configuration
├── data/                     # Local data landing zones, ignored by git except .gitkeep
├── dashboards/               # Dashboard definitions and terminal-facing views
├── docs/                     # Architecture, API, research, and operations documentation
├── infrastructure/           # Docker, CI, deployment, and environment assets
├── models/                   # Trained model artifacts, ignored as needed once generated
├── notebooks/                # Exploratory notebooks promoted into src/ when reusable
├── research/                 # Hypotheses, methodology, experiments, and reports by project
├── scripts/                  # Developer and research automation scripts
├── src/qr_haven/             # Installable Python package
└── tests/                    # Unit and integration tests
```

## Package Boundaries

- `data`: ingestion, validation, storage, and point-in-time access
- `features`: reusable factor and feature-store logic
- `research`: workflow orchestration and experiment lifecycle
- `alpha`: signal models and alpha validation
- `portfolio`: portfolio construction and optimizers
- `risk`: VaR, CVaR, stress, factor, liquidity, and drawdown analytics
- `costs`: transaction cost, borrow, financing, slippage, and market-impact models
- `execution`: execution algorithms and simulators
- `backtesting`: event/vectorized backtests and performance attribution
- `ml`: model training, evaluation, and tracking
- `ai_agents`: AI-native research automation
- `reporting`: reports, dashboards, and terminal payloads
- `integrations.market_terminal`: contracts for mounting QR-Haven inside `market_terminal`

## Promotion Rule

Research starts in `research/` or `notebooks/`. Once code becomes reusable across more than one
project, it moves into `src/qr_haven` with tests and typed interfaces.

## Project Documentation

Subsystem documentation should live in dedicated folders under `docs/`. The institutional
portfolio optimizer documentation begins in `docs/Institutional-Portfolio-Optimizer/`.
