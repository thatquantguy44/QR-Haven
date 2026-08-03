import math

import numpy as np
import pandas as pd
import pytest

from qr_haven.costs import (
    AlmgrenChrissModel,
    ImpactEstimate,
    SquareRootImpactModel,
    estimate_portfolio_impact,
    portfolio_total_impact_cost,
)


# ---------------------------------------------------------------------------
# ImpactEstimate
# ---------------------------------------------------------------------------


def test_impact_estimate_to_dict_contains_all_fields() -> None:
    est = ImpactEstimate(
        temporary_impact_bps=2.0,
        permanent_impact_bps=0.5,
        total_impact_bps=2.25,
    )
    data = est.to_dict()
    assert set(data) == {"temporary_impact_bps", "permanent_impact_bps", "total_impact_bps"}
    assert data["total_impact_bps"] == 2.25


# ---------------------------------------------------------------------------
# SquareRootImpactModel
# ---------------------------------------------------------------------------


def test_sqrt_model_known_value() -> None:
    # eta=0.1, sigma=0.02 (2% daily), participation=0.01 (1% of ADV)
    # impact_bps = 0.1 × 0.02 × 10_000 × sqrt(0.01) = 0.1 × 200 × 0.1 = 2.0
    model = SquareRootImpactModel(eta=0.1)
    est = model.estimate(participation_rate=0.01, daily_volatility=0.02)
    assert math.isclose(est.total_impact_bps, 2.0, rel_tol=1e-9)
    assert est.permanent_impact_bps == 0.0
    assert est.temporary_impact_bps == est.total_impact_bps


def test_sqrt_model_zero_participation_gives_zero_impact() -> None:
    est = SquareRootImpactModel().estimate(participation_rate=0.0, daily_volatility=0.02)
    assert est.total_impact_bps == 0.0


def test_sqrt_model_zero_volatility_gives_zero_impact() -> None:
    est = SquareRootImpactModel().estimate(participation_rate=0.05, daily_volatility=0.0)
    assert est.total_impact_bps == 0.0


def test_sqrt_model_impact_scales_with_sqrt_of_participation() -> None:
    model = SquareRootImpactModel(eta=0.1)
    est_1pct = model.estimate(0.01, 0.02)
    est_4pct = model.estimate(0.04, 0.02)
    assert math.isclose(est_4pct.total_impact_bps, est_1pct.total_impact_bps * 2.0, rel_tol=1e-9)


def test_sqrt_model_rejects_negative_participation() -> None:
    with pytest.raises(ValueError, match="participation_rate"):
        SquareRootImpactModel().estimate(-0.01, 0.02)


def test_sqrt_model_rejects_negative_volatility() -> None:
    with pytest.raises(ValueError, match="daily_volatility"):
        SquareRootImpactModel().estimate(0.01, -0.02)


def test_sqrt_model_rejects_non_positive_eta() -> None:
    with pytest.raises(ValueError, match="eta"):
        SquareRootImpactModel(eta=0.0)


# ---------------------------------------------------------------------------
# AlmgrenChrissModel
# ---------------------------------------------------------------------------


def test_ac_model_known_value() -> None:
    # eta=0.1, gamma=0.1, alpha=0.6, sigma=0.02, participation=0.01
    # sigma_bps = 200
    # temporary = 0.1 × 200 × 0.01^0.6
    # permanent = 0.1 × 200 × 0.01 = 0.2
    # total = temporary + 0.5 × 0.2 = temporary + 0.1
    model = AlmgrenChrissModel(eta=0.1, gamma=0.1, alpha=0.6)
    est = model.estimate(participation_rate=0.01, daily_volatility=0.02)
    expected_temporary = 0.1 * 200 * (0.01 ** 0.6)
    expected_permanent = 0.1 * 200 * 0.01
    assert math.isclose(est.temporary_impact_bps, expected_temporary, rel_tol=1e-9)
    assert math.isclose(est.permanent_impact_bps, expected_permanent, rel_tol=1e-9)
    assert math.isclose(
        est.total_impact_bps, expected_temporary + 0.5 * expected_permanent, rel_tol=1e-9
    )


def test_ac_model_alpha_half_matches_sqrt_model_without_permanent() -> None:
    # With gamma=0 and alpha=0.5, AC should equal the sqrt model
    ac = AlmgrenChrissModel(eta=0.1, gamma=0.0, alpha=0.5)
    sqrt_model = SquareRootImpactModel(eta=0.1)
    for prate in [0.005, 0.01, 0.05, 0.10]:
        ac_est = ac.estimate(prate, 0.02)
        sqrt_est = sqrt_model.estimate(prate, 0.02)
        assert math.isclose(ac_est.total_impact_bps, sqrt_est.total_impact_bps, rel_tol=1e-9)


def test_ac_model_zero_participation_gives_zero_impact() -> None:
    est = AlmgrenChrissModel().estimate(participation_rate=0.0, daily_volatility=0.02)
    assert est.total_impact_bps == 0.0


def test_ac_model_permanent_impact_is_linear_in_participation() -> None:
    # gamma drives permanent; permanent = gamma × sigma_bps × participation_rate
    model = AlmgrenChrissModel(eta=0.1, gamma=0.1, alpha=0.6)
    est_1 = model.estimate(0.01, 0.02)
    est_2 = model.estimate(0.02, 0.02)
    assert math.isclose(est_2.permanent_impact_bps, 2.0 * est_1.permanent_impact_bps, rel_tol=1e-9)


def test_ac_model_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="eta"):
        AlmgrenChrissModel(eta=0.0)
    with pytest.raises(ValueError, match="gamma"):
        AlmgrenChrissModel(gamma=-0.1)
    with pytest.raises(ValueError, match="alpha"):
        AlmgrenChrissModel(alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        AlmgrenChrissModel(alpha=1.1)


# ---------------------------------------------------------------------------
# Portfolio-level functions
# ---------------------------------------------------------------------------


def _portfolio_inputs() -> tuple[pd.Series, pd.Series, pd.Series]:
    symbols = ["AAPL", "MSFT", "GOOGL"]
    weight_changes = pd.Series([0.10, 0.05, 0.08], index=symbols)
    daily_vols = pd.Series([0.015, 0.018, 0.020], index=symbols)
    participation_rates = pd.Series([0.005, 0.010, 0.008], index=symbols)
    return weight_changes, daily_vols, participation_rates


def test_estimate_portfolio_impact_returns_correct_shape() -> None:
    wc, dv, pr = _portfolio_inputs()
    model = SquareRootImpactModel(eta=0.1)
    frame = estimate_portfolio_impact(wc, dv, pr, model)
    assert list(frame.index) == ["AAPL", "MSFT", "GOOGL"]
    assert "total_impact_bps" in frame.columns
    assert "weighted_cost" in frame.columns


def test_estimate_portfolio_impact_weighted_cost_matches_manual() -> None:
    symbols = ["AAPL"]
    wc = pd.Series([0.10], index=symbols)
    dv = pd.Series([0.02], index=symbols)
    pr = pd.Series([0.01], index=symbols)
    model = SquareRootImpactModel(eta=0.1)
    frame = estimate_portfolio_impact(wc, dv, pr, model)
    expected_impact_bps = 2.0  # from known-value test
    expected_weighted = expected_impact_bps * 0.10 / 10_000
    assert math.isclose(frame.loc["AAPL", "weighted_cost"], expected_weighted, rel_tol=1e-9)


def test_portfolio_total_impact_cost_sums_weighted_costs() -> None:
    wc, dv, pr = _portfolio_inputs()
    model = AlmgrenChrissModel()
    frame = estimate_portfolio_impact(wc, dv, pr, model)
    total = portfolio_total_impact_cost(frame)
    assert math.isclose(total, frame["weighted_cost"].sum(), rel_tol=1e-12)
    assert total > 0.0


def test_estimate_portfolio_impact_missing_symbol_in_vol_treated_as_zero() -> None:
    wc = pd.Series([0.10, 0.05], index=["AAPL", "UNKNOWN"])
    dv = pd.Series([0.02], index=["AAPL"])
    pr = pd.Series([0.01], index=["AAPL"])
    model = SquareRootImpactModel()
    frame = estimate_portfolio_impact(wc, dv, pr, model)
    assert frame.loc["UNKNOWN", "total_impact_bps"] == 0.0
