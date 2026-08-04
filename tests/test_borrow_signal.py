"""Tests for BorrowRateAlphaSignal."""

import numpy as np
import pandas as pd
import pytest

from qr_haven.alpha.borrow_signal import BorrowRateAlphaSignal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cross_section(n: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "borrow_rate": rng.uniform(0.005, 2.00, n),
            "borrow_utilization": rng.uniform(0.10, 0.99, n),
        }
    )


def _rate_only(n: int = 20, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"borrow_rate": rng.uniform(0.005, 2.00, n)})


def _panel(n_dates: int = 60, n_tickers: int = 15, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_dates, freq="B")
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    data = rng.uniform(0.005, 1.0, size=(n_dates, n_tickers))
    return pd.DataFrame(data, index=dates, columns=tickers)


# ---------------------------------------------------------------------------
# Init validation
# ---------------------------------------------------------------------------

class TestBorrowRateAlphaSignalInit:
    def test_default_direction(self):
        s = BorrowRateAlphaSignal()
        assert s.direction == "momentum"

    def test_contrarian_accepted(self):
        s = BorrowRateAlphaSignal(direction="contrarian")
        assert s.direction == "contrarian"

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError, match="direction"):
            BorrowRateAlphaSignal(direction="random")

    def test_utilization_weight_negative_raises(self):
        with pytest.raises(ValueError, match="utilization_weight"):
            BorrowRateAlphaSignal(utilization_weight=-0.1)

    def test_utilization_weight_one_raises(self):
        with pytest.raises(ValueError, match="utilization_weight"):
            BorrowRateAlphaSignal(utilization_weight=1.0)

    def test_zero_winsorize_raises(self):
        with pytest.raises(ValueError, match="winsorize_z"):
            BorrowRateAlphaSignal(winsorize_z=0.0)

    def test_utilization_weight_zero_accepted(self):
        s = BorrowRateAlphaSignal(utilization_weight=0.0)
        assert s.utilization_weight == 0.0


# ---------------------------------------------------------------------------
# compute() — cross-sectional
# ---------------------------------------------------------------------------

class TestComputeCrossSection:
    def test_missing_borrow_rate_raises(self):
        df = pd.DataFrame({"borrow_utilization": [0.5, 0.7, 0.9]})
        with pytest.raises(ValueError, match="borrow_rate"):
            BorrowRateAlphaSignal().compute(df)

    def test_output_shape(self):
        df = _cross_section()
        scores = BorrowRateAlphaSignal().compute(df)
        assert scores.shape == (len(df),)

    def test_output_series(self):
        df = _cross_section()
        scores = BorrowRateAlphaSignal().compute(df)
        assert isinstance(scores, pd.Series)

    def test_index_preserved(self):
        df = _cross_section().set_index(pd.Index([f"TK{i}" for i in range(20)]))
        scores = BorrowRateAlphaSignal().compute(df)
        assert list(scores.index) == list(df.index)

    def test_output_finite(self):
        df = _cross_section()
        scores = BorrowRateAlphaSignal().compute(df)
        assert np.all(np.isfinite(scores))

    def test_rate_only_df_works(self):
        df = _rate_only()
        scores = BorrowRateAlphaSignal().compute(df)
        assert len(scores) == len(df)

    def test_no_utilization_when_absent(self):
        df = _rate_only()
        s = BorrowRateAlphaSignal(include_utilization=True)
        scores = s.compute(df)
        assert len(scores) == len(df)


# ---------------------------------------------------------------------------
# Direction: momentum vs contrarian
# ---------------------------------------------------------------------------

class TestDirection:
    def _high_borrow_rank(self, scores: pd.Series, df: pd.DataFrame) -> float:
        """Rank correlation between borrow rate and alpha score."""
        return float(scores.rank().corr(df["borrow_rate"].rank()))

    def test_momentum_high_borrow_negative_alpha(self):
        df = _cross_section(n=30, seed=5)
        scores = BorrowRateAlphaSignal(direction="momentum").compute(df)
        corr = self._high_borrow_rank(scores, df)
        assert corr < -0.8

    def test_contrarian_high_borrow_positive_alpha(self):
        df = _cross_section(n=30, seed=5)
        scores = BorrowRateAlphaSignal(direction="contrarian").compute(df)
        corr = self._high_borrow_rank(scores, df)
        assert corr > 0.8

    def test_momentum_and_contrarian_are_negatives(self):
        df = _cross_section(n=20, seed=9)
        m_scores = BorrowRateAlphaSignal(direction="momentum").compute(df)
        c_scores = BorrowRateAlphaSignal(direction="contrarian").compute(df)
        pd.testing.assert_series_equal(-m_scores, c_scores, check_names=False)

    def test_highest_borrow_gets_most_negative_momentum(self):
        df = _cross_section(n=25, seed=3)
        scores = BorrowRateAlphaSignal(direction="momentum").compute(df)
        max_rate_idx = df["borrow_rate"].idxmax()
        assert scores[max_rate_idx] == scores.min()

    def test_highest_borrow_gets_most_positive_contrarian(self):
        df = _cross_section(n=25, seed=3)
        scores = BorrowRateAlphaSignal(direction="contrarian").compute(df)
        max_rate_idx = df["borrow_rate"].idxmax()
        assert scores[max_rate_idx] == scores.max()


# ---------------------------------------------------------------------------
# Utilization incorporation
# ---------------------------------------------------------------------------

class TestUtilization:
    def test_utilization_changes_scores(self):
        df = _cross_section(n=20, seed=7)
        s_no = BorrowRateAlphaSignal(include_utilization=False).compute(df)
        s_yes = BorrowRateAlphaSignal(include_utilization=True).compute(df)
        # with varied utilization the scores should differ
        assert not np.allclose(s_no.values, s_yes.values)

    def test_zero_utilization_weight_matches_rate_only(self):
        df = _cross_section(n=20, seed=7)
        s_zero_weight = BorrowRateAlphaSignal(
            include_utilization=True, utilization_weight=0.0
        ).compute(df)
        s_no_util = BorrowRateAlphaSignal(include_utilization=False).compute(df)
        pd.testing.assert_series_equal(s_zero_weight, s_no_util)

    def test_winsorisation_applied(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"borrow_rate": rng.uniform(0.01, 0.10, 18)})
        df = pd.concat(
            [df, pd.DataFrame({"borrow_rate": [1000.0, -1000.0]})],
            ignore_index=True,
        )
        scores = BorrowRateAlphaSignal(winsorize_z=2.0).compute(df)
        assert np.all(np.abs(scores) <= 2.0 + 1e-10)


# ---------------------------------------------------------------------------
# compute_panel()
# ---------------------------------------------------------------------------

class TestComputePanel:
    def test_output_shape(self):
        panel = _panel()
        result = BorrowRateAlphaSignal().compute_panel(panel)
        assert result.shape == panel.shape

    def test_output_finite_after_warmup(self):
        panel = _panel(n_dates=60, n_tickers=10)
        result = BorrowRateAlphaSignal().compute_panel(panel, change_window=20)
        # first 20 rows are NaN in change signal; drop and check rest
        valid = result.iloc[20:]
        assert np.all(np.isfinite(valid.values))

    def test_level_only(self):
        panel = _panel()
        result = BorrowRateAlphaSignal().compute_panel(
            panel, include_level=True, include_change=False
        )
        assert result.shape == panel.shape

    def test_change_only(self):
        panel = _panel()
        result = BorrowRateAlphaSignal().compute_panel(
            panel, include_level=False, include_change=True
        )
        assert result.shape == panel.shape

    def test_neither_level_nor_change_raises(self):
        panel = _panel()
        with pytest.raises(ValueError, match="At least one"):
            BorrowRateAlphaSignal().compute_panel(
                panel, include_level=False, include_change=False
            )

    def test_rising_borrow_bearish_in_momentum(self):
        """Ticker whose borrow rises monotonically should have negative change signal."""
        n_dates, n_tickers = 50, 5
        panel = _panel(n_dates=n_dates, n_tickers=n_tickers)
        # make ticker T00 have steadily rising borrow
        panel["T00"] = np.linspace(0.01, 2.0, n_dates)
        # make all others flat
        for col in panel.columns:
            if col != "T00":
                panel[col] = 0.10

        result = BorrowRateAlphaSignal(direction="momentum").compute_panel(
            panel, change_window=10, include_level=False, include_change=True
        )
        # after warmup, T00 change signal should be most negative
        later = result.iloc[15:]
        t00_mean = later["T00"].mean()
        others_mean = later.drop(columns=["T00"]).mean().mean()
        assert t00_mean < others_mean

    def test_rising_borrow_bearish_in_contrarian_change(self):
        """Change signal is always bearish regardless of direction setting."""
        panel = _panel()
        m_result = BorrowRateAlphaSignal(direction="momentum").compute_panel(
            panel, include_level=False, include_change=True
        )
        c_result = BorrowRateAlphaSignal(direction="contrarian").compute_panel(
            panel, include_level=False, include_change=True
        )
        pd.testing.assert_frame_equal(m_result, c_result)

    def test_index_and_columns_preserved(self):
        panel = _panel()
        result = BorrowRateAlphaSignal().compute_panel(panel)
        assert list(result.index) == list(panel.index)
        assert list(result.columns) == list(panel.columns)


# ---------------------------------------------------------------------------
# Uniform universe (edge case)
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_all_same_borrow_rate(self):
        df = pd.DataFrame({"borrow_rate": [0.05] * 10})
        scores = BorrowRateAlphaSignal().compute(df)
        np.testing.assert_allclose(scores.values, 0.0, atol=1e-10)

    def test_two_securities(self):
        df = pd.DataFrame({"borrow_rate": [0.01, 1.00]})
        scores = BorrowRateAlphaSignal(direction="contrarian").compute(df)
        assert scores.iloc[1] > scores.iloc[0]

    def test_name_of_output_series(self):
        df = _rate_only()
        scores = BorrowRateAlphaSignal().compute(df)
        assert scores.name == "borrow_alpha"
