"""Tests for ShortSqueezeModel."""

import numpy as np
import pandas as pd
import pytest

from qr_haven.costs.squeeze import ShortSqueezeModel, SqueezeScores

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_df(n: int = 20, seed: int = 0) -> pd.DataFrame:
    """Synthetic universe with all five squeeze factors."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "short_interest_ratio": rng.uniform(0.01, 0.60, n),
            "days_to_cover": rng.uniform(0.5, 20.0, n),
            "borrow_rate": rng.uniform(0.005, 1.50, n),
            "borrow_utilization": rng.uniform(0.10, 0.99, n),
            "price_return_20d": rng.uniform(-0.30, 0.30, n),
        }
    )


def _required_only_df(n: int = 20, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "short_interest_ratio": rng.uniform(0.01, 0.60, n),
            "days_to_cover": rng.uniform(0.5, 20.0, n),
            "borrow_rate": rng.uniform(0.005, 1.50, n),
        }
    )


def _squeezed_vs_safe(n_each: int = 30) -> pd.DataFrame:
    """Half safe (low metrics), half squeezed (high metrics)."""
    rng = np.random.default_rng(42)
    safe = pd.DataFrame(
        {
            "short_interest_ratio": rng.uniform(0.01, 0.05, n_each),
            "days_to_cover": rng.uniform(0.5, 2.0, n_each),
            "borrow_rate": rng.uniform(0.005, 0.02, n_each),
            "borrow_utilization": rng.uniform(0.10, 0.30, n_each),
            "price_return_20d": rng.uniform(-0.05, 0.0, n_each),
        }
    )
    squeezed = pd.DataFrame(
        {
            "short_interest_ratio": rng.uniform(0.40, 0.70, n_each),
            "days_to_cover": rng.uniform(15.0, 25.0, n_each),
            "borrow_rate": rng.uniform(0.50, 2.00, n_each),
            "borrow_utilization": rng.uniform(0.85, 0.99, n_each),
            "price_return_20d": rng.uniform(0.10, 0.30, n_each),
        }
    )
    return pd.concat([safe, squeezed], ignore_index=True)


# ---------------------------------------------------------------------------
# Init validation
# ---------------------------------------------------------------------------

class TestShortSqueezeModelInit:
    def test_default_init(self):
        m = ShortSqueezeModel()
        assert m.winsorize_z == 3.0
        assert m.min_universe == 5

    def test_custom_weights_accepted(self):
        w = {
            "short_interest_ratio": 0.5,
            "days_to_cover": 0.3,
            "borrow_rate": 0.2,
        }
        m = ShortSqueezeModel(weights=w)
        assert m.weights["short_interest_ratio"] == 0.5

    def test_missing_required_weight_raises(self):
        with pytest.raises(ValueError, match="required factors"):
            ShortSqueezeModel(weights={"short_interest_ratio": 1.0, "days_to_cover": 0.5})

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            ShortSqueezeModel(
                weights={
                    "short_interest_ratio": -0.1,
                    "days_to_cover": 0.5,
                    "borrow_rate": 0.6,
                }
            )

    def test_bad_tier_thresholds_raises(self):
        with pytest.raises(ValueError, match="tier_thresholds"):
            ShortSqueezeModel(tier_thresholds=(0.7, 0.3))

    def test_tier_thresholds_equal_raises(self):
        with pytest.raises(ValueError):
            ShortSqueezeModel(tier_thresholds=(0.5, 0.5))

    def test_zero_winsorize_raises(self):
        with pytest.raises(ValueError, match="winsorize_z"):
            ShortSqueezeModel(winsorize_z=0.0)

    def test_min_universe_one_raises(self):
        with pytest.raises(ValueError, match="min_universe"):
            ShortSqueezeModel(min_universe=1)


# ---------------------------------------------------------------------------
# score() input validation
# ---------------------------------------------------------------------------

class TestScoreValidation:
    def test_missing_required_column_raises(self):
        df = _base_df()
        df = df.drop(columns=["borrow_rate"])
        with pytest.raises(ValueError, match="Missing required columns"):
            ShortSqueezeModel().score(df)

    def test_too_small_universe_raises(self):
        df = _base_df(n=4)
        with pytest.raises(ValueError, match="min_universe"):
            ShortSqueezeModel().score(df)

    def test_accepts_minimum_universe(self):
        df = _base_df(n=5)
        result = ShortSqueezeModel().score(df)
        assert len(result.squeeze_score) == 5

    def test_accepts_required_only_columns(self):
        df = _required_only_df()
        result = ShortSqueezeModel().score(df)
        assert len(result.squeeze_score) == len(df)


# ---------------------------------------------------------------------------
# Output shape and contracts
# ---------------------------------------------------------------------------

class TestScoreOutput:
    def test_squeeze_score_shape(self):
        df = _base_df()
        result = ShortSqueezeModel().score(df)
        assert isinstance(result, SqueezeScores)
        assert len(result.squeeze_score) == len(df)

    def test_squeeze_score_in_unit_interval(self):
        df = _base_df(n=50)
        result = ShortSqueezeModel().score(df)
        assert np.all(result.squeeze_score >= 0.0)
        assert np.all(result.squeeze_score <= 1.0)

    def test_squeeze_tier_labels(self):
        df = _base_df(n=50)
        result = ShortSqueezeModel().score(df)
        valid = {"LOW", "MODERATE", "HIGH"}
        assert set(result.squeeze_tier.unique()).issubset(valid)

    def test_component_zscores_keys(self):
        df = _base_df()
        result = ShortSqueezeModel().score(df)
        # all five factors are in the input
        expected = {
            "short_interest_ratio", "days_to_cover", "borrow_rate",
            "borrow_utilization", "price_return_20d",
        }
        assert expected.issubset(result.component_zscores.keys())

    def test_component_zscores_winsorised(self):
        df = _base_df()
        result = ShortSqueezeModel(winsorize_z=2.0).score(df)
        for z in result.component_zscores.values():
            assert np.all(np.abs(z) <= 2.0 + 1e-10)

    def test_index_preserved(self):
        df = _base_df().set_index(pd.Index([f"T{i}" for i in range(20)]))
        result = ShortSqueezeModel().score(df)
        assert list(result.squeeze_score.index) == list(df.index)

    def test_optional_factor_absent_does_not_raise(self):
        df = _required_only_df()
        result = ShortSqueezeModel().score(df)
        assert "borrow_utilization" not in result.component_zscores


# ---------------------------------------------------------------------------
# Tier assignment consistency
# ---------------------------------------------------------------------------

class TestTierAssignment:
    def test_tier_consistent_with_score(self):
        df = _base_df(n=50)
        m = ShortSqueezeModel(tier_thresholds=(0.33, 0.67))
        result = m.score(df)
        low_mask = result.squeeze_score < 0.33
        high_mask = result.squeeze_score >= 0.67
        assert (result.squeeze_tier[low_mask] == "LOW").all()
        assert (result.squeeze_tier[high_mask] == "HIGH").all()
        between = (~low_mask) & (~high_mask)
        assert (result.squeeze_tier[between] == "MODERATE").all()

    def test_custom_thresholds(self):
        df = _base_df(n=30)
        m = ShortSqueezeModel(tier_thresholds=(0.20, 0.80))
        result = m.score(df)
        assert (result.squeeze_tier[result.squeeze_score < 0.20] == "LOW").all()
        assert (result.squeeze_tier[result.squeeze_score >= 0.80] == "HIGH").all()


# ---------------------------------------------------------------------------
# Economic directional test
# ---------------------------------------------------------------------------

class TestDirectionality:
    def test_squeezed_stocks_score_higher_than_safe(self):
        df = _squeezed_vs_safe(n_each=30)
        result = ShortSqueezeModel().score(df)
        safe_mean = result.squeeze_score.iloc[:30].mean()
        squeezed_mean = result.squeeze_score.iloc[30:].mean()
        assert squeezed_mean > safe_mean

    def test_squeezed_mostly_high_tier(self):
        df = _squeezed_vs_safe(n_each=30)
        result = ShortSqueezeModel().score(df)
        squeezed_tiers = result.squeeze_tier.iloc[30:]
        high_frac = (squeezed_tiers == "HIGH").mean()
        assert high_frac > 0.5

    def test_safe_mostly_low_tier(self):
        df = _squeezed_vs_safe(n_each=30)
        result = ShortSqueezeModel().score(df)
        safe_tiers = result.squeeze_tier.iloc[:30]
        low_frac = (safe_tiers == "LOW").mean()
        assert low_frac > 0.5


# ---------------------------------------------------------------------------
# score_to_frame
# ---------------------------------------------------------------------------

class TestScoreToFrame:
    def test_returns_dataframe(self):
        df = _base_df()
        out = ShortSqueezeModel().score_to_frame(df)
        assert isinstance(out, pd.DataFrame)

    def test_original_columns_preserved(self):
        df = _base_df()
        out = ShortSqueezeModel().score_to_frame(df)
        for col in df.columns:
            assert col in out.columns

    def test_squeeze_score_column_added(self):
        df = _base_df()
        out = ShortSqueezeModel().score_to_frame(df)
        assert "squeeze_score" in out.columns

    def test_squeeze_tier_column_added(self):
        df = _base_df()
        out = ShortSqueezeModel().score_to_frame(df)
        assert "squeeze_tier" in out.columns

    def test_z_columns_added(self):
        df = _base_df()
        out = ShortSqueezeModel().score_to_frame(df)
        assert "borrow_rate_z" in out.columns
        assert "days_to_cover_z" in out.columns

    def test_same_scores_as_score_method(self):
        df = _base_df()
        m = ShortSqueezeModel()
        direct = m.score(df).squeeze_score
        frame = m.score_to_frame(df)["squeeze_score"]
        pd.testing.assert_series_equal(direct, frame)


# ---------------------------------------------------------------------------
# Custom weights
# ---------------------------------------------------------------------------

class TestCustomWeights:
    def test_borrow_rate_only_weights(self):
        """With all weight on borrow_rate, ranking should follow borrow_rate."""
        w = {
            "short_interest_ratio": 0.0,
            "days_to_cover": 0.0,
            "borrow_rate": 1.0,
        }
        df = _base_df()
        result = ShortSqueezeModel(weights=w).score(df)
        # score rank should correlate with borrow_rate rank
        rank_score = result.squeeze_score.rank()
        rank_rate = df["borrow_rate"].rank()
        corr = float(rank_score.corr(rank_rate))
        assert corr > 0.9

    def test_zero_weight_factor_excluded_from_zscores(self):
        w = {
            "short_interest_ratio": 0.0,
            "days_to_cover": 0.5,
            "borrow_rate": 0.5,
        }
        df = _base_df()
        result = ShortSqueezeModel(weights=w).score(df)
        assert "short_interest_ratio" not in result.component_zscores


# ---------------------------------------------------------------------------
# Uniform universe (edge case)
# ---------------------------------------------------------------------------

class TestUniformUniverse:
    def test_all_same_borrow_rate(self):
        """Constant factor → zero z-score → all scores equal."""
        df = _base_df()
        df["borrow_rate"] = 0.05
        df["borrow_utilization"] = 0.50
        df["price_return_20d"] = 0.0
        # only short_interest_ratio and days_to_cover vary
        result = ShortSqueezeModel().score(df)
        # scores should still be valid
        assert np.all(result.squeeze_score >= 0.0)
        assert np.all(result.squeeze_score <= 1.0)
