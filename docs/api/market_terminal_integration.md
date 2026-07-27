# market_terminal Integration

QR-Haven should expose research capabilities to the separate `market_terminal` project through
stable Python contracts rather than direct UI coupling.

The first integration surface is:

- `TerminalPanel`: a serializable panel payload for tables, charts, diagnostics, and reports
- `TerminalPlugin`: a protocol for QR-Haven modules that publish one or more terminal panels
- `TerminalPluginRegistry`: a local registry until `market_terminal` owns discovery and mounting

Initial terminal candidates:

- Portfolio optimizer diagnostics
- Risk summary and stress panels
- Alpha signal quality dashboards
- Execution and transaction cost analytics
- Regime detection status

