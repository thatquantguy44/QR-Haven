"""Real-time inference functions for the borrow demand surface.

These functions are called at query time (intraday) to support operational
decisions: shortage detection, rate confirmation, and allocation triggers.

Reference: spec § Phase 3 — Downstream Applications.
"""

from __future__ import annotations

import torch
import gpytorch

from qr_haven.borrow_demand.features import SurfaceFeatures
from qr_haven.borrow_demand.model import BorrowDemandSurface


def shortage_probability(
    model: BorrowDemandSurface,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    features: SurfaceFeatures,
    inventory_shares: float,
) -> float:
    """P(D(s,t) > inventory_shares | features) from the posterior predictive CDF.

    Uses the normal complementary CDF (norm.sf) against the GP posterior
    predictive mean and standard deviation at *features*.

    Parameters
    ----------
    model : BorrowDemandSurface
        Fitted (eval-mode) SVGP model.
    likelihood : GaussianLikelihood
        Fitted likelihood with learned noise variance.
    features : SurfaceFeatures
        Normalised characteristic features for the (security, time) query point.
        Must include trading_time_norm when model.use_time_kernel is True.
    inventory_shares : float
        Available-to-lend shares for the security at the time of the query.

    Returns
    -------
    float in [0, 1] — probability that true demand exceeds available inventory.
    A value > 0.5 suggests likely shortage; > 0.9 warrants proactive restriction.

    Notes
    -----
    The posterior is Gaussian, so the shortage probability is:
        P(D > C) = 1 - Φ((C - μ_post) / σ_post)
                 = norm.sf(C, loc=μ_post, scale=σ_post)
    where C = inventory_shares, μ_post and σ_post are the GP posterior
    predictive mean and std at the query point.
    """
    from scipy.stats import norm

    use_time = getattr(model, "use_time_kernel", False)
    x = (features.to_full_tensor() if use_time else features.to_tensor()).unsqueeze(0)

    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(x))

    mu = float(pred.mean.item())
    sigma = float(pred.variance.sqrt().item())

    if sigma <= 0.0:
        return 0.0 if mu <= inventory_shares else 1.0

    return float(norm.sf(inventory_shares, loc=mu, scale=sigma))


def demand_quantile(
    model: BorrowDemandSurface,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    features: SurfaceFeatures,
    quantile: float = 0.95,
) -> float:
    """Return the q-th quantile of the posterior demand distribution at *features*.

    Useful for conservative inventory reservation: set aside demand_quantile(q=0.95)
    shares to cover all but the top-5% demand scenarios.

    Parameters
    ----------
    quantile : float
        Probability level in (0, 1).  Default 0.95 (95th percentile).

    Returns
    -------
    float — demand in shares at the requested quantile.
    """
    from scipy.stats import norm

    if not (0.0 < quantile < 1.0):
        raise ValueError(f"quantile must be in (0, 1); got {quantile}")

    use_time = getattr(model, "use_time_kernel", False)
    x = (features.to_full_tensor() if use_time else features.to_tensor()).unsqueeze(0)

    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(x))

    mu = float(pred.mean.item())
    sigma = float(pred.variance.sqrt().item())

    return float(norm.ppf(quantile, loc=mu, scale=max(sigma, 1e-12)))
