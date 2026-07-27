import numpy as np
import pandas as pd

from qr_haven.integrations.market_terminal import risk_metrics_panel
from qr_haven.risk import SimpleRiskEngine, portfolio_returns


def test_portfolio_returns_aligns_weights_to_return_columns() -> None:
    returns = pd.DataFrame(
        {
            "AAPL": [0.01, -0.02, 0.03],
            "MSFT": [0.02, 0.01, -0.01],
        },
        index=pd.date_range("2024-01-01", periods=3, tz="UTC"),
    )
    weights = pd.Series({"MSFT": 0.25, "AAPL": 0.75})

    result = portfolio_returns(weights, returns)

    expected = pd.Series(
        [0.0125, -0.0125, 0.02],
        index=returns.index,
        name="portfolio_return",
    )
    pd.testing.assert_series_equal(result, expected)


def test_simple_risk_engine_returns_core_metrics() -> None:
    returns = pd.DataFrame(
        {
            "AAPL": [0.01, -0.02, 0.03, -0.01],
            "MSFT": [0.02, 0.01, -0.01, 0.00],
        },
        index=pd.date_range("2024-01-01", periods=4, tz="UTC"),
    )
    weights = pd.Series({"AAPL": 0.6, "MSFT": 0.4})

    metrics = SimpleRiskEngine(annualization=252, var_confidence=0.95).evaluate(
        weights,
        returns,
    )

    assert metrics["observations"] == 4
    assert np.isclose(metrics["gross_exposure"], 1.0)
    assert np.isclose(metrics["net_exposure"], 1.0)
    assert np.isclose(metrics["concentration"], 0.52)
    assert np.isclose(metrics["largest_weight"], 0.6)
    assert metrics["annualized_volatility"] > 0.0
    assert metrics["max_drawdown"] <= 0.0
    assert metrics["value_at_risk"] > 0.0
    assert metrics["conditional_value_at_risk"] >= metrics["value_at_risk"]


def test_simple_risk_engine_handles_zero_volatility() -> None:
    returns = pd.DataFrame(
        {"AAPL": [0.01, 0.01, 0.01]},
        index=pd.date_range("2024-01-01", periods=3, tz="UTC"),
    )
    weights = pd.Series({"AAPL": 1.0})

    metrics = SimpleRiskEngine().evaluate(weights, returns)

    assert metrics["volatility"] == 0.0
    assert metrics["sharpe_ratio"] == 0.0


def test_risk_metrics_panel_returns_terminal_table_payload() -> None:
    metrics = {"volatility": 0.10, "sharpe_ratio": 1.25}

    panel = risk_metrics_panel(metrics)

    assert panel.panel_id == "risk.metrics"
    assert panel.kind == "table"
    assert panel.payload["rows"] == [
        {"metric": "volatility", "value": 0.10},
        {"metric": "sharpe_ratio", "value": 1.25},
    ]
