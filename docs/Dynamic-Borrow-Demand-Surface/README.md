# Dynamic Borrow Demand Surface

**Priority Score: 98 / 100**

> Model borrow demand as a continuously evolving surface over security characteristics
> rather than independently per ticker. A Dynamic Gaussian Process learns
> D(s, t) = f(σ, ADV, SI, ETF, RQ) so that shortages are identified earlier,
> inventory is priced correctly, and locate allocation decisions are data-driven.

---

## Problem Statement

The traditional approach to securities lending demand forecasting treats each name
independently: a model is fit per ticker using that ticker's own history.  This fails
in three systematic ways:

1. **Cold-start problem** — a newly listed name or one that has never been HTB has
   no demand history. The desk prices it at generic rates and misses early squeeze signals.

2. **Cross-sectional information loss** — borrow demand co-moves across names with
   similar characteristics (high short interest, low float, concentrated ETF ownership).
   Per-ticker models cannot exploit this structure.

3. **Lag in rate discovery** — by the time a per-ticker model accumulates enough data
   to detect a tightening, the borrow has already become scarce and the desk is either
   over-allocated or pricing below market.

The **Dynamic Borrow Demand Surface** treats demand as a latent function over a
low-dimensional characteristic space:

```
D(s, t) = f(σ_s, ADV_s, SI_s, ETF_s, RQ_s, t)
```

where the surface evolves continuously in time.  Querying the surface for any
(security, time) pair returns a posterior distribution over demand, enabling
probabilistic shortage detection, principled rate-setting, and proactive locate allocation.

---

## Mathematical Formulation

### Feature Space

Each security s at time t is embedded into a characteristic vector:

```
x(s, t) = [σ_s(t),  ADV_s(t),  SI_s(t),  ETF_s(t),  RQ_s(t)]
```

| Feature | Symbol | Description |
|---|---|---|
| Realised / implied volatility | σ | 20-day realised vol or nearest-term IV; proxy for squeeze probability |
| Average daily volume | ADV | 20-day median; normalises demand by tradeable float |
| Short interest ratio | SI | Shares sold short / float; direct demand proxy at weekly cadence |
| ETF ownership concentration | ETF | Fraction of float held in ETFs; governs create/redeem pressure |
| Recent locate request quantity | RQ | Rolling 5-day locate volume; real-time demand signal |

All features are normalised to [0, 1] via rolling percentile rank across the
cross-section, making the surface comparable across different market regimes.

### Demand Surface as a Gaussian Process

Place a GP prior over the latent demand function:

```
D ~ GP(μ(x), k(x, x'))
```

**Mean function** `μ(x)`:  affine in characteristics, fit via OLS on historical
loan data as a warm start.

**Kernel** `k(x, x')`:  composite of three components:

```
k(x, x') = k_char(x, x') + k_time(t, t') + k_noise
```

| Component | Form | Purpose |
|---|---|---|
| `k_char` | ARD Matérn-5/2 over [σ, ADV, SI, ETF, RQ] | Smoothly varying demand across characteristics |
| `k_time` | Ornstein-Uhlenbeck over t | Exponential forgetting of stale observations |
| `k_noise` | Diagonal nugget ε²I | Observation noise (request volume is noisy) |

ARD (Automatic Relevance Determination) lengthscales ℓ_j are learned by marginal
likelihood maximisation, automatically down-weighting uninformative features.

### Posterior Inference

Given n observations `{(x_i, t_i, d_i)}` (historical loan/locate request pairs):

```
D(x*, t*) | data ~ N(μ_post(x*, t*), σ²_post(x*, t*))
```

```
μ_post = μ(x*) + K(x*, X) [K(X, X) + σ²_n I]⁻¹ (d - μ(X))
σ²_post = k(x*, x*) - K(x*, X) [K(X, X) + σ²_n I]⁻¹ K(X, x*)
```

The posterior mean is the point estimate of demand; the posterior variance is
used for shortfall probability and pricing uncertainty bounds.

### Scalability: Sparse GP with Inducing Points

Full GP inference is O(n³).  For n > 10,000 daily observations (realistic for a
multi-name lending book) use the **Sparse Variational GP** (SVGP) approximation
(Titsias 2009, Hensman et al. 2013):

- Select m ≪ n **inducing points** Z in feature space (m ~ 500 covers the surface well)
- Variational posterior q(u) over inducing outputs u = f(Z)
- Training cost O(nm²); prediction cost O(m) per query

The inducing points are placed by k-means++ on the normalised feature matrix and
refined via gradient descent during hyperparameter optimisation.

### Dynamic (Online) Updates

The temporal kernel `k_time = σ²_f · exp(−|t − t'| / ℓ_t)` implements exponential
decay with lengthscale ℓ_t (≈ 5–10 trading days).  New observations received
intraday are incorporated via a **streaming posterior update**:

```
# Rank-1 Kalman update when a new observation (x_new, d_new) arrives
K_new = K(X_new, Z)             # (1, m) cross-covariance
alpha = L⁻¹ K_new.T             # m-vector
mu_post += alpha * (d_new - mu_prior(x_new))
Sigma_post -= alpha alpha.T / (sigma_n² + k(x_new, x_new) - alpha.T alpha)
```

This keeps the posterior current without re-running full batch inference.

### Alternative: Neural Operator Learning

For desks with sufficient data (> 100k locate request events), a **Fourier Neural
Operator** (FNO; Li et al. 2021) can replace the GP as the surface model:

```
D(·, t+1) = F_θ(D(·, t), x(·, t))
```

where F_θ is a learned operator mapping the current demand surface and
characteristic cross-section to the next period's surface.  The FNO operates
in Fourier space over the characteristic dimensions, enabling O(m log m) inference.

**When to prefer FNO over GP**:

| Criterion | GP | FNO |
|---|---|---|
| Data volume | < 50k observations | > 100k observations |
| Uncertainty quantification | Native posterior | Requires MC dropout / conformal |
| Cold-start generalisation | Strong (kernel encodes prior) | Weaker (data-hungry) |
| Intraday latency | < 1 ms per query | ~5–50 ms per batch |
| Training cost | Minutes | Hours (GPU) |

The spec focuses on GP as the primary model; FNO is the extension path.

---

## Problem Hierarchy

| Sub-problem | Formulation | Solver / Method |
|---|---|---|
| Hyperparameter learning | Marginal log-likelihood maximisation | L-BFGS-B (scipy) or Adam |
| Sparse GP posterior | SVGP ELBO maximisation | GPyTorch + inducing points |
| Online update | Streaming rank-1 Kalman step | NumPy; < 1 ms |
| Shortage detection | P(D > inventory) > threshold | Posterior CDF query |
| Rate recommendation | E[borrow fee | D] from demand-rate mapping | Monotone spline calibration |
| Locate allocation | Argmax expected revenue subject to supply cap | Greedy / LP post-processing |

---

## Suggested Implementation

### Phase 1 — Offline Surface Estimation

**Goal**: fit and validate the GP surface on historical data; establish baselines.

```python
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

class BorrowDemandSurface(ApproximateGP):
    """Sparse variational GP over security characteristics."""

    def __init__(self, inducing_points: torch.Tensor) -> None:
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.LinearMean(input_size=inducing_points.size(1))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=inducing_points.size(1))
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
```

**Training loop** (ELBO maximisation):
```python
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO

likelihood = GaussianLikelihood()
model = BorrowDemandSurface(inducing_points=Z_init)

optimizer = torch.optim.Adam([
    {"params": model.parameters()},
    {"params": likelihood.parameters()},
], lr=1e-2)
mll = VariationalELBO(likelihood, model, num_data=len(train_loader.dataset))

for epoch in range(n_epochs):
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(x_batch)
        loss = -mll(output, y_batch)
        loss.backward()
        optimizer.step()
```

**Feature construction**:
```python
import numpy as np
from dataclasses import dataclass

@dataclass
class SurfaceFeatures:
    """Normalised cross-sectional features for the demand surface."""
    sigma_pct: float      # Percentile rank of 20-day realised vol in cross-section
    adv_pct: float        # Percentile rank of 20-day ADV in cross-section
    si_pct: float         # Percentile rank of short interest ratio
    etf_pct: float        # Percentile rank of ETF ownership fraction
    rq_pct: float         # Percentile rank of 5-day rolling locate request volume

    @property
    def tensor(self) -> np.ndarray:
        return np.array([self.sigma_pct, self.adv_pct, self.si_pct,
                         self.etf_pct, self.rq_pct], dtype=np.float32)
```

### Phase 2 — Online Update Engine

The online updater runs as a lightweight intraday service, consuming locate request
events and applying streaming posterior updates without re-fitting the full model:

```python
class OnlineSurfaceUpdater:
    """Streaming rank-1 posterior updates for the borrow demand surface."""

    def __init__(
        self,
        model: BorrowDemandSurface,
        likelihood: GaussianLikelihood,
        sigma_n: float,
    ) -> None:
        self.model = model
        self.sigma_n = sigma_n
        # Cached posterior state (inducing mean / covariance)
        self._mu = None
        self._Sigma = None

    def update(self, x_new: np.ndarray, d_new: float) -> None:
        """Incorporate a single new locate request observation."""
        x_t = torch.tensor(x_new, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            K_new = self.model.covar_module(x_t, self.model.variational_strategy.inducing_points)
            K_new = K_new.evaluate().squeeze(0).numpy()
        if self._mu is None:
            return  # Batch fit not yet available
        alpha = np.linalg.solve(self._Sigma, K_new)
        residual = d_new - float(self.model.mean_module(x_t).item())
        noise_var = self.sigma_n ** 2 + float(
            self.model.covar_module(x_t).evaluate().item()
        ) - float(K_new @ alpha)
        self._mu += alpha * residual / noise_var
        self._Sigma -= np.outer(alpha, alpha) / noise_var
```

### Phase 3 — Downstream Applications

#### Shortage Probability

```python
def shortage_probability(
    surface: BorrowDemandSurface,
    features: SurfaceFeatures,
    inventory: float,
    likelihood: GaussianLikelihood,
) -> float:
    """P(D(s,t) > inventory | features) from the posterior predictive CDF."""
    x = torch.tensor(features.tensor).unsqueeze(0)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(surface(x))
    mu = pred.mean.item()
    sigma = pred.variance.sqrt().item()
    from scipy.stats import norm
    return float(norm.sf(inventory, loc=mu, scale=sigma))
```

#### Rate Recommendation

A monotone spline calibration maps posterior demand quantiles to borrow fees:

```
fee(s, t) = g(D_50(s,t), D_95(s,t))
```

where D_50 / D_95 are the posterior median and 95th-percentile demand estimates.
The spline g is fit by isotonic regression on (historical demand, realised borrow fee)
pairs, guaranteeing that higher demand always maps to higher fees.

#### Locate Allocation

Given a supply cap `C_s` per name and a set of pending locate requests ranked by
posterior expected demand, the allocation problem is:

```
max  Σ_i  fee_i · z_i
s.t. Σ_{i: name(i)=s}  z_i · qty_i ≤ C_s   ∀ s
     z_i ∈ {0, 1}
```

This is a standard binary knapsack per name, solved greedily in O(n log n) by
sorting requests by fee/qty ratio within each name.

---

## Data Requirements

### Minimum Viable vs. Full Dataset

| Dataset | Minimum Viable | Full Production |
|---|---|---|
| Locate request history | 6 months, ≥ 500 names | 24 months, all lendable names |
| Loan transaction history | 6 months | 36 months (for rate calibration) |
| Short interest | Weekly cadence, same 500 names | Daily Markit, full cross-section |
| ETF holdings | Top-50 ETFs by AUM | All ETFs with lendable constituents |
| Market data (vol, ADV) | Daily close, 6-month lookback | Intraday OHLCV, 36-month lookback |
| Inventory | Daily snapshot | Intraday tick from stock record |

---

### Dataset 1 — Locate Request Log

**Source**: Prime brokerage workflow system (e.g. Broadridge, SunGard).
**Cadence**: Intraday tick; events arrive in real time.
**Join key**: `cusip` + `locate_date` links to all daily datasets.

| Field | Type | Unit / Format | Required | Notes |
|---|---|---|---|---|
| `locate_id` | `str` | UUID or internal ID | Yes | Deduplication key |
| `event_timestamp` | `datetime` | UTC, microsecond precision | Yes | When request was received |
| `locate_date` | `date` | YYYY-MM-DD | Yes | Calendar date; used as join key to daily feeds |
| `cusip` | `str(9)` | CUSIP-9 | Yes | Primary security identifier |
| `isin` | `str(12)` | ISO 6166 | No | Fallback identifier |
| `ticker` | `str` | Exchange ticker | No | Human reference only; not a join key |
| `client_id` | `str` | Internal counterparty code | Yes | For allocation logic; do not use raw name |
| `requested_qty_shares` | `int` | Number of shares | Yes | Raw requested size |
| `approved_qty_shares` | `int` | Number of shares | Yes | 0 if fully rejected |
| `fee_bps` | `float` | Annualised basis points | Yes | Fee at time of approval; 0 if rejected |
| `status` | `enum` | `approved / partial / rejected` | Yes | Rejection reasons drive cold-start signal |
| `rejection_reason` | `str` | Free text / code | No | `NO_INVENTORY`, `HTB`, `COUNTERPARTY_LIMIT` |
| `expiry_timestamp` | `datetime` | UTC | Yes | When the locate expires |
| `settlement_date` | `date` | YYYY-MM-DD | Yes | T+2 standard; T+1 for some ETFs |

**Derived target variable** (what the model predicts):

```
demand_obs(cusip, locate_date) = sum(requested_qty_shares)
                                 grouped by (cusip, locate_date)
```

Use `requested_qty_shares` (not `approved_qty_shares`) as the demand signal — approved quantity is supply-constrained and will understate true demand on HTB names.

**Missing data handling**: If `fee_bps` is missing on an approved locate, impute from the same-day rate card for that CUSIP.  If rate card is also missing, flag the row and exclude from rate calibration but retain for demand estimation.

---

### Dataset 2 — Loan Transaction Log

**Source**: Securities lending desk order management system.
**Cadence**: Intraday tick; end-of-day reconciled file also required.
**Join key**: `cusip` + `trade_date`.

| Field | Type | Unit / Format | Required | Notes |
|---|---|---|---|---|
| `loan_id` | `str` | UUID or trade ID | Yes | Deduplication key |
| `trade_date` | `date` | YYYY-MM-DD | Yes | Origination date |
| `settlement_date` | `date` | YYYY-MM-DD | Yes | Actual settlement date |
| `cusip` | `str(9)` | CUSIP-9 | Yes | |
| `counterparty_id` | `str` | LEI or internal code | Yes | |
| `direction` | `enum` | `lend / borrow` | Yes | From the desk's perspective |
| `quantity_shares` | `int` | Number of shares | Yes | |
| `notional_usd` | `float` | USD | Yes | `quantity_shares × price_at_trade` |
| `collateral_type` | `enum` | `cash / non-cash` | Yes | Affects margin cost calculation |
| `rebate_rate_bps` | `float` | Annualised bps | Yes | Negative = desk pays; used for fee surface |
| `fee_bps` | `float` | Annualised bps | Yes | `rebate_rate_bps` from the lender's perspective |
| `term_type` | `enum` | `open / term` | Yes | Open loans reprice daily |
| `term_end_date` | `date` | YYYY-MM-DD | No | Only for `term` loans |
| `recall_date` | `date` | YYYY-MM-DD | No | Populated when lender recalls |
| `status` | `enum` | `open / closed / recalled` | Yes | |

**Used for**: (1) rate spline calibration target `(demand → fee_bps)`; (2) computing `on_loan_shares` in inventory; (3) historical shortage labels (`fill_rate < 1` on a given CUSIP-date).

---

### Dataset 3 — Short Interest

**Source (primary)**: IHS Markit Daily Short Sale Data (subscription required).
**Source (fallback)**: FINRA short interest (bi-monthly, free, T+2 lag).
**Cadence**: Daily (Markit); bi-monthly (FINRA).
**Join key**: `cusip` + `as_of_date`.

| Field | Type | Unit / Format | Required | Notes |
|---|---|---|---|---|
| `as_of_date` | `date` | YYYY-MM-DD | Yes | Markit = T-1 settle; FINRA = mid/end of month |
| `cusip` | `str(9)` | CUSIP-9 | Yes | |
| `isin` | `str(12)` | ISO 6166 | No | Cross-reference |
| `shares_short` | `int` | Number of shares | Yes | Reported short position |
| `shares_float` | `int` | Number of shares | Yes | Public float (not total shares outstanding) |
| `si_ratio` | `float` | Decimal (0–1+) | Yes | `shares_short / shares_float`; cap at 2.0 for outlier treatment |
| `days_to_cover` | `float` | Trading days | Yes | `shares_short / avg_daily_volume_20d` |
| `short_exempt_shares` | `int` | Number of shares | No | Market-maker exemptions; exclude from SI calc if present |
| `utilisation_rate` | `float` | Decimal (0–1) | Markit only | `on_loan_shares / lendable_shares`; strongest demand signal |
| `data_source` | `enum` | `markit / finra / estimated` | Yes | Used to apply appropriate lag adjustment |

**Lag handling**: Markit SI reflects T-1 borrow activity; FINRA reflects T-14 average.
When joining to the feature matrix, use:
```
si_for_date(t) = markit_si(t-1)           # if Markit available
               = finra_si(most_recent)    # otherwise, with staleness flag
```

Staleness flag: if `as_of_date < t - 5 trading days`, set `si_stale = True` and
increase the SI feature's uncertainty (widen the input noise in the GP kernel for
that observation).

---

### Dataset 4 — ETF Holdings

**Source**: ETF issuer daily holdings files (most large ETFs publish these; available
via Bloomberg ETF data or direct from iShares/SPDR/Vanguard portals).
**Cadence**: Daily, published before market open (reflects previous close).
**Join key**: `component_cusip` + `as_of_date`.

| Field | Type | Unit / Format | Required | Notes |
|---|---|---|---|---|
| `as_of_date` | `date` | YYYY-MM-DD | Yes | Holdings effective date |
| `etf_cusip` | `str(9)` | CUSIP-9 | Yes | The ETF itself |
| `etf_ticker` | `str` | Exchange ticker | Yes | Human reference |
| `etf_aum_usd` | `float` | USD millions | Yes | Used to weight concentration score |
| `component_cusip` | `str(9)` | CUSIP-9 | Yes | The underlying security |
| `shares_held` | `int` | Number of shares | Yes | ETF's position in the component |
| `nav_weight` | `float` | Decimal (0–1) | Yes | Component weight in ETF NAV |
| `creation_unit_size` | `int` | Shares | Yes | Minimum creation/redemption basket size |
| `in_kind_flag` | `bool` | Boolean | Yes | Whether component is delivered in-kind on create/redeem |

**Derived feature — ETF concentration** (what goes into the surface):
```
etf_ownership_pct(cusip, date) =
    sum(shares_held × in_kind_flag)          # shares held across all ETFs
    / shares_float(cusip, date)              # from short interest dataset

etf_score(cusip, date) =
    sum(etf_aum_usd × nav_weight × in_kind_flag)   # AUM-weighted coverage
```

Use `etf_ownership_pct` as the raw feature; `etf_score` as a secondary diagnostic.
Only include ETFs with `in_kind_flag = True` — cash-settled ETFs do not create
create/redeem pressure on the underlying borrow market.

---

### Dataset 5 — Market Data (Price, Volume, Volatility)

**Source**: Internal market data warehouse or vendor (Bloomberg, Refinitiv).
**Cadence**: Daily close (minimum); intraday OHLCV preferred for vol computation.
**Join key**: `cusip` + `trade_date`.

| Field | Type | Unit / Format | Required | Notes |
|---|---|---|---|---|
| `trade_date` | `date` | YYYY-MM-DD | Yes | |
| `cusip` | `str(9)` | CUSIP-9 | Yes | |
| `ticker` | `str` | Exchange ticker | No | Human reference only |
| `close_price_usd` | `float` | USD per share | Yes | Unadjusted close; use adjusted for return calculation |
| `adj_close_price_usd` | `float` | USD per share | Yes | Split- and dividend-adjusted |
| `volume_shares` | `int` | Number of shares | Yes | Total exchange volume (all venues) |
| `dollar_volume_usd` | `float` | USD | Yes | `volume_shares × vwap`; preferred over close-price-based |
| `vwap_usd` | `float` | USD per share | No | Volume-weighted average price |
| `realized_vol_20d` | `float` | Annualised decimal | Yes | See formula below |
| `realized_vol_5d` | `float` | Annualised decimal | No | Short-window supplement |
| `iv_atm_1m` | `float` | Annualised decimal | No | Nearest-term ATM implied vol; fallback to `realized_vol_20d` |
| `iv_source` | `enum` | `options / estimated` | Yes | Whether IV came from options chain or was imputed |
| `adv_20d_shares` | `float` | Shares | Yes | 20-day median daily share volume |
| `adv_20d_usd` | `float` | USD | Yes | 20-day median daily dollar volume; primary ADV signal |
| `market_cap_usd` | `float` | USD millions | No | Used in cross-sectional normalisation |

**Exact formula — realised volatility**:
```
log_returns(t) = log(adj_close(t) / adj_close(t-1))
realized_vol_20d(t) = std(log_returns(t-19 : t)) × sqrt(252)   # annualised
```
Use 20 trading-day (not calendar-day) lookback.  Exclude days with zero volume (halts).
Minimum 15 non-zero return observations required; otherwise mark `realized_vol_20d = NaN`
and use the cross-sectional median as imputation.

**Exact formula — ADV**:
```
adv_20d_usd(t) = median(dollar_volume_usd(t-19 : t))   # median, not mean
```
Median is preferred to mean to resist earnings-day volume spikes that would
inflate ADV for low-liquidity names.

---

### Dataset 6 — Inventory (Available to Lend)

**Source**: Stock record system (prime brokerage internal ledger).
**Cadence**: Intraday snapshots at market open, midday, close; also EOD reconciled.
**Join key**: `cusip` + `as_of_datetime`.

| Field | Type | Unit / Format | Required | Notes |
|---|---|---|---|---|
| `as_of_datetime` | `datetime` | UTC | Yes | Snapshot timestamp |
| `cusip` | `str(9)` | CUSIP-9 | Yes | |
| `available_to_lend_shares` | `int` | Shares | Yes | Net lendable after existing loans and reserves |
| `on_loan_shares` | `int` | Shares | Yes | Currently lent |
| `total_lendable_shares` | `int` | Shares | Yes | `available_to_lend + on_loan` |
| `reserved_shares` | `int` | Shares | Yes | Pledged but not yet settled |
| `htb_flag` | `bool` | Boolean | Yes | Hard-to-borrow designation; set by desk |
| `gc_flag` | `bool` | Boolean | Yes | General collateral (easy borrow); complement of HTB |
| `recall_risk_shares` | `int` | Shares | No | Shares at risk of being recalled by lenders |
| `utilisation_rate` | `float` | Decimal (0–1) | Yes | `on_loan / total_lendable`; key shortage predictor |

**Shortage label construction** (used for backtesting evaluation):
```
shortage_event(cusip, date) = True
    if utilisation_rate(cusip, date) > 0.95
    AND available_to_lend_shares(cusip, date) < median_daily_locate_qty(cusip)
```

---

### Dataset 7 — Realised Borrow Fees

**Source**: Lending desk PnL system; may be extracted from loan transaction log (Dataset 2).
**Cadence**: Daily.
**Join key**: `cusip` + `trade_date`.

| Field | Type | Unit / Format | Required | Notes |
|---|---|---|---|---|
| `trade_date` | `date` | YYYY-MM-DD | Yes | |
| `cusip` | `str(9)` | CUSIP-9 | Yes | |
| `fee_bps_median` | `float` | Annualised bps | Yes | Median fee across all loans for this CUSIP on this date |
| `fee_bps_p25` | `float` | Annualised bps | Yes | 25th-percentile fee; lower bound for rate calibration |
| `fee_bps_p75` | `float` | Annualised bps | Yes | 75th-percentile fee |
| `fee_bps_p95` | `float` | Annualised bps | Yes | 95th-percentile; HTB rate ceiling |
| `total_loans` | `int` | Count | Yes | Number of loans; used to weight the spline calibration |
| `total_notional_usd` | `float` | USD | Yes | Weighted sum; larger loans should dominate calibration |

This dataset is the **calibration target** for the monotone spline `g: demand → fee`.
Use `fee_bps_median` weighted by `total_notional_usd` for the spline fit.

---

### Join Keys and Data Lineage

```
Locate Request Log ─┐
                    ├──► [cusip + locate_date] ──► Feature Matrix ──► Surface Model
Loan Transaction ───┤
Short Interest ─────┤           ▲ joins on:
ETF Holdings ───────┤           │  cusip → component_cusip (ETF)
Market Data ────────┤           │  trade_date / as_of_date / locate_date
Inventory ──────────┘

Feature Matrix columns (one row per cusip-date):
  cusip, date,
  demand_shares (target),              ← from Locate Request Log
  si_ratio, utilisation_rate,          ← from Short Interest
  etf_ownership_pct,                   ← from ETF Holdings
  realized_vol_20d, iv_atm_1m,         ← from Market Data
  adv_20d_usd,                         ← from Market Data
  rq_5d_shares,                        ← rolling 5-day sum of demand_shares
  available_to_lend_shares,            ← from Inventory
  fee_bps_median                        ← from Realised Borrow Fees (label for calibration)
```

---

### Feature Construction Pipeline

All raw fields must be transformed to normalised percentile ranks before entering
the GP.  The pipeline runs once per trading day after all daily closes are received.

```
Step 1 — Compute raw features per (cusip, date):
  sigma_raw   = realized_vol_20d if iv_source == 'options' then iv_atm_1m else realized_vol_20d
  adv_raw     = adv_20d_usd
  si_raw      = si_ratio  (with lag adjustment; see Dataset 3)
  etf_raw     = etf_ownership_pct
  rq_raw      = sum(requested_qty_shares, last 5 locate_dates)

Step 2 — Cross-sectional percentile rank (within the universe on date t):
  sigma_pct   = percentile_rank(sigma_raw,  universe_t)   → [0, 1]
  adv_pct     = percentile_rank(adv_raw,    universe_t)   → [0, 1]
  si_pct      = percentile_rank(si_raw,     universe_t)   → [0, 1]
  etf_pct     = percentile_rank(etf_raw,    universe_t)   → [0, 1]
  rq_pct      = percentile_rank(rq_raw,     universe_t)   → [0, 1]

Step 3 — Missing value imputation:
  If sigma_raw is NaN      → sigma_pct = 0.5  (cross-sectional median)
  If si_raw is stale       → si_pct = last_known_si_pct × 0.95  (decay factor)
  If etf_raw is missing    → etf_pct = 0.0  (assume no ETF pressure)
  If rq_raw is 0           → rq_pct = 0.0  (no recent demand observed)

Step 4 — Construct SurfaceFeatures:
  x(s, t) = [sigma_pct, adv_pct, si_pct, etf_pct, rq_pct]  ∈ [0,1]^5
```

**Universe definition**: include any security where `total_lendable_shares > 0` on
date t and the security has been active (non-zero close volume) for ≥ 60 consecutive
trading days.  Exclude ADRs, preferred shares, and ETFs themselves (they appear only
as `component_cusip`, not as lending names).

---

### Data Quality Checks

Run these assertions before each training run and flag failures:

| Check | Condition | Action on Failure |
|---|---|---|
| Locate log completeness | `approved_qty + rejected_qty = requested_qty` per row | Log warning; keep row |
| Fee sign | `fee_bps ≥ 0` for all approved locates | Clamp to 0; flag for review |
| SI ratio bounds | `0 ≤ si_ratio ≤ 2.0` | Clamp; values > 2 indicate data error |
| Vol non-negative | `realized_vol_20d > 0` | Replace with cross-sectional median |
| ADV minimum | `adv_20d_usd > 100_000` (USD) | Exclude from universe (illiquid) |
| ETF weight sum | `sum(nav_weight per ETF) ≈ 1.0 ± 0.01` | Flag ETF file as stale |
| Inventory non-negative | `available_to_lend_shares ≥ 0` | Clamp; alert stock record team |
| Locate date coverage | ≥ 200 distinct CUSIP-dates per calendar month | Abort training if below threshold |

---

### Minimum Viable Dataset Summary

| Field | Minimum | Rationale |
|---|---|---|
| Locate request history | 6 months, ≥ 500 names | Enough cross-sectional variation for GP hyperparameter learning |
| Loan transaction history | 6 months | Rate spline needs ≥ 100 HTB observations per demand decile |
| Short interest | Weekly cadence, same 500 names | SI is the strongest single feature; cannot be omitted |
| ETF holdings | Top-50 ETFs by AUM | Covers ~85% of ETF-driven borrow pressure |
| Market data | Daily close, 6-month lookback | 20-day vol and ADV require ≥ 20 prior trading days |
| Inventory | Daily EOD snapshot | Intraday not required for offline training |
| Realised borrow fees | 3 months, ≥ 50 HTB names | Spline calibration needs HTB observations; GC names alone insufficient |

---

## Baseline Models

| Baseline | Description | Known Weakness |
|---|---|---|
| **Kernel regression** | Nadaraya-Watson over (SI, σ) with Gaussian kernel | No temporal dynamics; bandwidth selection is ad-hoc |
| **Per-ticker ARIMA** | AR(5) on each name's daily locate volume | Cold-start failure; no cross-sectional transfer |
| **Simple heuristic** | HTB flag triggered by SI > 20% | Binary; ignores ADV, ETF, RQ; no probability output |
| **Linear panel model** | Fixed-effects regression D_{s,t} = α_s + β x_{s,t} + ε | Cannot capture nonlinear interactions; no uncertainty |

---

## Evaluation Metrics

| Metric | Formula | Target |
|---|---|---|
| **Surface RMSE** | √(E[(D̂ − D)²]) on held-out names | < 15% of mean demand |
| **Shortage recall** | TP / (TP + FN) at P(D > C) > 0.7 | > 80% of shortfalls detected ≥ 1 day early |
| **Calibration** (reliability) | |P̂(D > C) − empirical freq| < 5% per decile | Posterior is well-calibrated |
| **Economic PnL** | (Fee from model allocation − fee from desk heuristic) / notional | > 5 bps annualised improvement |
| **Inventory utilisation** | Fraction of available inventory lent at any time | > 90% peak utilisation |
| **Rate accuracy** | MAE(fee_model − fee_actual) | < 20 bps |

Backtesting protocol: **rolling-origin cross-validation** — fit on months 1–T, evaluate
on month T+1, advance by one month.  Prevents look-ahead on SI and fee data.

---

## Integration Points

| System | Interface | Direction |
|---|---|---|
| Securities lending workflow | gRPC locate request feed | Inbound (demand observations) |
| Stock record / inventory system | REST poll or Kafka topic | Inbound (supply) |
| Rate card / pricing engine | REST endpoint `GET /surface/rate?cusip=&qty=&settle=` | Outbound |
| Locate approval workflow | REST endpoint `POST /locate/approve` returning `{approved, fee}` | Outbound |
| Risk dashboard | Websocket push of shortage probabilities | Outbound |
| Synthetic Inventory Optimizer | `baseline_borrow_bps` from surface rate endpoint | Outbound — feeds the HTB baseline cost |

---

## Module Layout

```
src/qr_haven/borrow_demand/
├── __init__.py
├── features.py          # SurfaceFeatures dataclass; cross-sectional normalisation
├── model.py             # BorrowDemandSurface (SVGP); BorrowDemandConfig
├── trainer.py           # Offline ELBO training loop; hyperparameter search
├── updater.py           # OnlineSurfaceUpdater; streaming rank-1 Kalman step
├── calibration.py       # Monotone spline: demand quantile → borrow fee
├── allocator.py         # Locate allocator; binary knapsack per name
└── diagnostics.py       # Calibration plots; surface visualisation; RMSE tables
```

---

## Complexity and Feasibility

| Dimension | Assessment |
|---|---|
| **Difficulty** | Medium-High |
| **Estimated ROI** | Very High |
| **Publication Value** | ★★★★★ |
| **Production Feasibility** | High |
| **Primary risk** | Data availability — locate request logs must be extracted from workflow system; SI data may have T+2 lag that limits intraday utility |
| **Secondary risk** | GP scaling — SVGP with m=500 inducing points is manageable; names with sparse history require strong kernel priors |
| **Mitigation** | Start with offline weekly surface; add intraday streaming update in Phase 2 after data pipeline is established |

---

## References

- Titsias, M. K. (2009). *Variational learning of inducing variables in sparse Gaussian processes.* AISTATS.
- Hensman, J., Fusi, N., & Lawrence, N. D. (2013). *Gaussian processes for big data.* UAI.
- Li, Z., et al. (2021). *Fourier neural operator for parametric partial differential equations.* ICLR.
- GPyTorch: Gardner et al. (2018). *GPyTorch: Blackbox matrix-matrix Gaussian process inference with GPU acceleration.* NeurIPS.
- D'Avolio, G. (2002). *The market for borrowing stock.* Journal of Financial Economics 66(2-3), 271–306. (Foundational empirical work on borrow demand drivers.)
- Cohen, L., Diether, K., & Malloy, C. (2007). *Supply and demand shifts in the shorting market.* Journal of Finance 62(5), 2061–2096.
