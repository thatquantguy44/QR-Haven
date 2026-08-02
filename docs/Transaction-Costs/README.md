# Transaction Cost Models

This documentation folder covers QR-Haven's market impact and transaction cost models.

Accurate transaction cost modeling is essential for realistic backtesting. A strategy
that looks profitable gross of costs can be unprofitable net of impact, especially at
higher frequencies or with lower-liquidity names. The models here are designed to slot
into the QR-Haven P&L attribution waterfall:

```text
gross return
  → market impact costs   ← this module
  → borrow costs (short side)
  → financing costs
  → lending revenue (long side)
  → net return
```

## Models

### Square-Root Impact Model

The industry-standard workhorse. Impact scales with the square root of the participation
rate (trade size relative to average daily volume):

```text
impact_bps = eta × sigma_daily_bps × sqrt(participation_rate)
```

### Almgren-Chriss Model

Separates temporary and permanent impact components. Temporary impact is the price
concession paid to attract liquidity; permanent impact is the lasting shift in
equilibrium price from revealed order flow:

```text
temporary_bps = eta × sigma_daily_bps × participation_rate^alpha
permanent_bps = gamma × sigma_daily_bps × participation_rate
total_bps     = temporary_bps + 0.5 × permanent_bps
```

## Documentation Map

- [Market Impact Models](market_impact.md)
- [API: Market Impact](../api/market_impact.md)
