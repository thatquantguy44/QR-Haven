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

| Input | Source | Update Cadence | Notes |
|---|---|---|---|
| Loan request log | Securities lending desk | Intraday (tick) | Timestamp, CUSIP, quantity, counterparty |
| Locate request log | Prime brokerage workflow | Intraday (tick) | Approved / rejected flag needed |
| Short interest ratio | FINRA / IHS Markit | Weekly (bi-weekly public) | Per CUSIP, lagged ~T+2 |
| ETF holdings | ETF issuer / Bloomberg | Daily | Map security → ETF basket weight |
| Realised volatility | Market data feed | Daily close | 20-day trailing, annualised |
| Implied volatility | Options chain | Daily close | Nearest-term ATM IV; fallback to realised |
| ADV | Market data feed | Daily close | 20-day median dollar volume |
| Inventory levels | Stock record system | Intraday | Per-CUSIP available-to-lend |
| Realised borrow fees | Lending desk PnL | Daily | For spline calibration of rate surface |

**Minimum viable dataset**: 6 months of locate request logs + daily SI + daily ADV
across ≥ 500 names.  Surface quality improves substantially with 24 months of history.

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
