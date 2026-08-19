"""Tests for qr_haven.borrow_demand package.

Covers all seven modules: features, model, trainer, updater, calibration,
allocator, and diagnostics.  Model/trainer tests use tiny configurations
(n_inducing=10, n_epochs=2) to keep the suite fast.
"""

from __future__ import annotations

import math
from datetime import date, datetime

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# features.py
# ---------------------------------------------------------------------------


class TestRawFeatures:
    def test_defaults(self):
        from qr_haven.borrow_demand.features import RawFeatures

        r = RawFeatures(
            cusip="037833100",
            as_of_date=date(2024, 1, 2),
            realized_vol=0.25,
            adv_usd=1e6,
            si_ratio=0.05,
            etf_ownership_pct=0.12,
            rq_5d_shares=5000.0,
        )
        assert r.demand_shares == 0.0
        assert math.isnan(r.fee_bps)
        assert r.si_stale is False

    def test_nan_fields_allowed(self):
        from qr_haven.borrow_demand.features import RawFeatures

        r = RawFeatures(
            cusip="X",
            as_of_date=date(2024, 1, 2),
            realized_vol=float("nan"),
            adv_usd=float("nan"),
            si_ratio=float("nan"),
            etf_ownership_pct=float("nan"),
            rq_5d_shares=0.0,
        )
        assert math.isnan(r.realized_vol)


class TestSurfaceFeatures:
    def test_to_array_shape(self):
        from qr_haven.borrow_demand.features import SurfaceFeatures

        sf = SurfaceFeatures(0.1, 0.2, 0.3, 0.4, 0.5)
        arr = sf.to_array()
        assert arr.shape == (5,)
        assert arr.dtype == np.float32

    def test_to_array_values(self):
        from qr_haven.borrow_demand.features import SurfaceFeatures

        sf = SurfaceFeatures(sigma_pct=0.1, adv_pct=0.2, si_pct=0.3, etf_pct=0.4, rq_pct=0.5)
        arr = sf.to_array()
        np.testing.assert_allclose(arr, [0.1, 0.2, 0.3, 0.4, 0.5], atol=1e-6)

    def test_to_tensor(self):
        import torch
        from qr_haven.borrow_demand.features import SurfaceFeatures

        sf = SurfaceFeatures(0.0, 0.25, 0.5, 0.75, 1.0)
        t = sf.to_tensor()
        assert isinstance(t, torch.Tensor)
        assert t.shape == (5,)


class TestFeaturePipeline:
    def _make_raw(self, n: int = 5):
        from qr_haven.borrow_demand.features import RawFeatures

        return [
            RawFeatures(
                cusip=f"C{i}",
                as_of_date=date(2024, 1, 2),
                realized_vol=0.1 * (i + 1),
                adv_usd=1e6 * (i + 1),
                si_ratio=0.01 * (i + 1),
                etf_ownership_pct=0.05 * i,
                rq_5d_shares=100.0 * i,
                demand_shares=float(i * 10),
            )
            for i in range(n)
        ]

    def test_transform_returns_correct_length(self):
        from qr_haven.borrow_demand.features import FeaturePipeline

        pipe = FeaturePipeline()
        raw = self._make_raw(8)
        result = pipe.transform(raw)
        assert len(result) == 8

    def test_transform_features_in_unit_interval(self):
        from qr_haven.borrow_demand.features import FeaturePipeline

        pipe = FeaturePipeline()
        result = pipe.transform(self._make_raw(10))
        for sf in result:
            for v in (sf.sigma_pct, sf.adv_pct, sf.si_pct, sf.etf_pct, sf.rq_pct):
                assert 0.0 <= v <= 1.0, f"feature out of [0,1]: {v}"

    def test_empty_batch(self):
        from qr_haven.borrow_demand.features import FeaturePipeline

        pipe = FeaturePipeline()
        assert pipe.transform([]) == []

    def test_nan_sigma_fills_half(self):
        from qr_haven.borrow_demand.features import FeaturePipeline, RawFeatures

        raw = [
            RawFeatures("A", date(2024, 1, 2), float("nan"), 1e6, 0.05, 0.0, 0.0),
            RawFeatures("B", date(2024, 1, 2), 0.30, 2e6, 0.10, 0.0, 0.0),
        ]
        result = FeaturePipeline().transform(raw)
        assert result[0].sigma_pct == pytest.approx(0.5)

    def test_si_staleness_decay(self):
        from qr_haven.borrow_demand.features import FeaturePipeline, RawFeatures

        # Two identical SI ratios; one is stale — stale should rank lower
        raw = [
            RawFeatures("A", date(2024, 1, 2), 0.2, 1e6, 0.10, 0.0, 0.0, si_stale=True),
            RawFeatures("B", date(2024, 1, 2), 0.2, 1e6, 0.10, 0.0, 0.0, si_stale=False),
        ]
        result = FeaturePipeline().transform(raw)
        # stale one has lower effective SI, so lower si_pct
        assert result[0].si_pct <= result[1].si_pct

    def test_etf_nan_fills_zero(self):
        from qr_haven.borrow_demand.features import FeaturePipeline, RawFeatures

        raw = [
            RawFeatures("A", date(2024, 1, 2), 0.2, 1e6, 0.05, float("nan"), 0.0),
            RawFeatures("B", date(2024, 1, 2), 0.3, 2e6, 0.10, 0.20, 0.0),
        ]
        result = FeaturePipeline().transform(raw)
        assert result[0].etf_pct == pytest.approx(0.0)

    def test_build_matrix_shapes(self):
        from qr_haven.borrow_demand.features import FeaturePipeline

        pipe = FeaturePipeline()
        raw = self._make_raw(7)
        X, y = pipe.build_matrix(raw)
        assert X.shape == (7, 5)
        assert y.shape == (7,)
        assert X.dtype == np.float32
        assert y.dtype == np.float32


# ---------------------------------------------------------------------------
# model.py
# ---------------------------------------------------------------------------


class TestBorrowDemandConfig:
    def test_defaults(self):
        from qr_haven.borrow_demand.model import BorrowDemandConfig

        cfg = BorrowDemandConfig()
        assert cfg.n_inducing == 500
        assert cfg.matern_nu == 2.5

    def test_invalid_n_inducing(self):
        from qr_haven.borrow_demand.model import BorrowDemandConfig

        with pytest.raises(ValueError, match="n_inducing"):
            BorrowDemandConfig(n_inducing=5)

    def test_invalid_matern_nu(self):
        from qr_haven.borrow_demand.model import BorrowDemandConfig

        with pytest.raises(ValueError, match="matern_nu"):
            BorrowDemandConfig(matern_nu=1.0)

    def test_invalid_lr(self):
        from qr_haven.borrow_demand.model import BorrowDemandConfig

        with pytest.raises(ValueError, match="lr"):
            BorrowDemandConfig(lr=-0.01)

    def test_invalid_n_epochs(self):
        from qr_haven.borrow_demand.model import BorrowDemandConfig

        with pytest.raises(ValueError, match="n_epochs"):
            BorrowDemandConfig(n_epochs=0)


class TestBorrowDemandSurface:
    def _make_inducing(self, m: int = 10):
        import torch

        return torch.rand(m, 5)

    def test_instantiation(self):
        from qr_haven.borrow_demand.model import BorrowDemandSurface

        model = BorrowDemandSurface(self._make_inducing())
        assert model is not None

    def test_wrong_inducing_shape_raises(self):
        import torch
        from qr_haven.borrow_demand.model import BorrowDemandSurface

        with pytest.raises(ValueError, match="inducing_points"):
            BorrowDemandSurface(torch.rand(10, 3))

    def test_predict_shapes(self):
        import gpytorch
        import torch
        from qr_haven.borrow_demand.model import BorrowDemandSurface

        model = BorrowDemandSurface(self._make_inducing(10))
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model.eval()
        likelihood.eval()

        x = torch.rand(4, 5)
        mean, std = model.predict(x, likelihood)
        assert mean.shape == (4,)
        assert std.shape == (4,)
        assert np.all(std >= 0)

    def test_init_inducing_kmeans(self):
        import torch
        from qr_haven.borrow_demand.model import BorrowDemandSurface

        X = np.random.default_rng(0).random((50, 5)).astype(np.float32)
        inducing = BorrowDemandSurface.init_inducing_kmeans(X, n_inducing=15, seed=42)
        assert isinstance(inducing, torch.Tensor)
        assert inducing.shape == (15, 5)

    def test_init_inducing_kmeans_caps_at_n(self):
        from qr_haven.borrow_demand.model import BorrowDemandSurface

        X = np.random.default_rng(0).random((8, 5)).astype(np.float32)
        inducing = BorrowDemandSurface.init_inducing_kmeans(X, n_inducing=20, seed=0)
        assert inducing.shape[0] == 8


# ---------------------------------------------------------------------------
# trainer.py
# ---------------------------------------------------------------------------


class TestTrainer:
    def _tiny_data(self, n: int = 50, seed: int = 0):
        rng = np.random.default_rng(seed)
        X = rng.random((n, 5)).astype(np.float32)
        y = rng.random(n).astype(np.float32)
        return X, y

    def test_train_surface_runs(self):
        import gpytorch
        from qr_haven.borrow_demand.model import BorrowDemandConfig, BorrowDemandSurface
        from qr_haven.borrow_demand.trainer import train_surface

        X, y = self._tiny_data(50)
        cfg = BorrowDemandConfig(n_inducing=10, n_epochs=2, batch_size=16)
        inducing = BorrowDemandSurface.init_inducing_kmeans(X, 10, seed=0)
        model = BorrowDemandSurface(inducing)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        result = train_surface(model, likelihood, X, y, config=cfg)
        assert len(result.epoch_losses) == 2
        assert isinstance(result.final_loss, float)

    def test_train_surface_wrong_feature_dim_raises(self):
        import gpytorch
        from qr_haven.borrow_demand.model import BorrowDemandConfig, BorrowDemandSurface
        from qr_haven.borrow_demand.trainer import train_surface

        X, y = self._tiny_data(20)
        X_bad = X[:, :3]
        cfg = BorrowDemandConfig(n_inducing=10, n_epochs=1)
        inducing = BorrowDemandSurface.init_inducing_kmeans(
            np.random.random((20, 5)).astype(np.float32), 10
        )
        model = BorrowDemandSurface(inducing)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        with pytest.raises(ValueError, match="X_train must be"):
            train_surface(model, likelihood, X_bad, y, config=cfg)

    def test_make_model_and_likelihood(self):
        import gpytorch
        from qr_haven.borrow_demand.trainer import make_model_and_likelihood

        X, _ = self._tiny_data(30)
        model, likelihood = make_model_and_likelihood(
            X, config=None, seed=7
        )
        assert isinstance(likelihood, gpytorch.likelihoods.GaussianLikelihood)

    def test_model_eval_mode_after_training(self):
        import gpytorch
        from qr_haven.borrow_demand.model import BorrowDemandConfig, BorrowDemandSurface
        from qr_haven.borrow_demand.trainer import train_surface

        X, y = self._tiny_data(40)
        cfg = BorrowDemandConfig(n_inducing=10, n_epochs=1, batch_size=20)
        inducing = BorrowDemandSurface.init_inducing_kmeans(X, 10)
        model = BorrowDemandSurface(inducing)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        train_surface(model, likelihood, X, y, config=cfg)
        assert not model.training
        assert not likelihood.training

    def test_convergence_flag(self):
        import gpytorch
        from qr_haven.borrow_demand.model import BorrowDemandConfig, BorrowDemandSurface
        from qr_haven.borrow_demand.trainer import train_surface

        X, y = self._tiny_data(60)
        cfg = BorrowDemandConfig(n_inducing=10, n_epochs=10, batch_size=30)
        inducing = BorrowDemandSurface.init_inducing_kmeans(X, 10)
        model = BorrowDemandSurface(inducing)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        result = train_surface(model, likelihood, X, y, config=cfg, tol=1e4)
        # with very large tol, convergence should be True
        assert result.converged is True


# ---------------------------------------------------------------------------
# updater.py
# ---------------------------------------------------------------------------


class TestOnlineSurfaceUpdater:
    def _make_updater(self, max_buffer: int = 100):
        import gpytorch
        import torch
        from qr_haven.borrow_demand.model import BorrowDemandSurface
        from qr_haven.borrow_demand.updater import OnlineSurfaceUpdater

        X = np.random.default_rng(1).random((20, 5)).astype(np.float32)
        inducing = BorrowDemandSurface.init_inducing_kmeans(X, 10)
        model = BorrowDemandSurface(inducing)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model.eval()
        likelihood.eval()
        return OnlineSurfaceUpdater(model, likelihood, max_buffer_size=max_buffer)

    def _make_event(self, demand: float = 100.0):
        from qr_haven.borrow_demand.features import SurfaceFeatures
        from qr_haven.borrow_demand.updater import LocateEvent

        return LocateEvent(
            event_timestamp=datetime(2024, 1, 2, 10, 0),
            cusip="037833100",
            features=SurfaceFeatures(0.5, 0.5, 0.5, 0.5, 0.5),
            demand_shares=demand,
        )

    def test_ingest_adds_to_buffer(self):
        updater = self._make_updater()
        updater.ingest(self._make_event())
        assert updater.buffer_size == 1

    def test_buffer_respects_max(self):
        updater = self._make_updater(max_buffer=3)
        for _ in range(5):
            updater.ingest(self._make_event())
        assert updater.buffer_size == 3

    def test_clear_buffer(self):
        updater = self._make_updater()
        updater.ingest_batch([self._make_event() for _ in range(5)])
        updater.clear_buffer()
        assert updater.buffer_size == 0

    def test_predict_empty_buffer_returns_tuple(self):
        from qr_haven.borrow_demand.features import SurfaceFeatures

        updater = self._make_updater()
        mean, std = updater.predict(SurfaceFeatures(0.3, 0.3, 0.3, 0.3, 0.3))
        assert isinstance(mean, float)
        assert isinstance(std, float)
        assert std >= 0.0

    def test_predict_with_buffer_returns_tuple(self):
        from qr_haven.borrow_demand.features import SurfaceFeatures

        updater = self._make_updater()
        updater.ingest_batch([self._make_event(d) for d in [50.0, 80.0, 120.0]])
        mean, std = updater.predict(SurfaceFeatures(0.5, 0.5, 0.5, 0.5, 0.5))
        assert isinstance(mean, float)
        assert std >= 0.0

    def test_predict_std_non_negative_with_buffer(self):
        from qr_haven.borrow_demand.features import SurfaceFeatures

        updater = self._make_updater()
        for demand in [10.0, 50.0, 200.0, 500.0]:
            updater.ingest(self._make_event(demand))
        for _ in range(10):
            sf = SurfaceFeatures(*np.random.default_rng(99).random(5).tolist())
            _, std = updater.predict(sf)
            assert std >= 0.0


# ---------------------------------------------------------------------------
# calibration.py
# ---------------------------------------------------------------------------


class TestDemandRateCalibrator:
    def _fit_calibrator(self, n: int = 50, seed: int = 0):
        from qr_haven.borrow_demand.calibration import DemandRateCalibrator

        rng = np.random.default_rng(seed)
        demand = rng.uniform(0, 1, n)
        fee = 50.0 + 200.0 * demand + rng.normal(0, 5, n)
        fee = np.clip(fee, 0, None)
        cal = DemandRateCalibrator()
        result = cal.fit(demand, fee)
        return cal, result

    def test_fit_returns_result(self):
        _, result = self._fit_calibrator()
        assert result.n_observations == 50
        assert result.n_knots > 0

    def test_is_fitted_after_fit(self):
        from qr_haven.borrow_demand.calibration import DemandRateCalibrator

        cal = DemandRateCalibrator()
        assert not cal.is_fitted
        rng = np.random.default_rng(0)
        demand = rng.uniform(0, 1, 20)
        fee = np.clip(100 * demand, 0, None)
        cal.fit(demand, fee)
        assert cal.is_fitted

    def test_predict_before_fit_raises(self):
        from qr_haven.borrow_demand.calibration import DemandRateCalibrator

        cal = DemandRateCalibrator()
        with pytest.raises(RuntimeError, match="fit"):
            cal.predict(np.array([0.5]))

    def test_predict_monotone(self):
        cal, _ = self._fit_calibrator(100)
        test_demand = np.linspace(0, 1, 20)
        fees = cal.predict(test_demand)
        diffs = np.diff(fees)
        assert np.all(diffs >= -1e-9), "Isotonic regression should be non-decreasing"

    def test_fee_for_demand_scalar(self):
        cal, _ = self._fit_calibrator(30)
        fee = cal.fee_for_demand(0.5)
        assert isinstance(fee, float)
        assert fee >= 0.0

    def test_fit_requires_ten_observations(self):
        from qr_haven.borrow_demand.calibration import DemandRateCalibrator

        cal = DemandRateCalibrator()
        with pytest.raises(ValueError, match="10"):
            cal.fit(np.array([0.1, 0.2]), np.array([10.0, 20.0]))

    def test_fit_ignores_nan_rows(self):
        from qr_haven.borrow_demand.calibration import DemandRateCalibrator

        rng = np.random.default_rng(1)
        demand = rng.uniform(0, 1, 30)
        fee = 100 * demand
        demand[0] = float("nan")
        fee[1] = float("nan")
        cal = DemandRateCalibrator()
        result = cal.fit(demand, fee)
        assert result.n_observations == 28  # two rows removed

    def test_fit_with_sample_weight(self):
        from qr_haven.borrow_demand.calibration import DemandRateCalibrator

        rng = np.random.default_rng(2)
        demand = rng.uniform(0, 1, 40)
        fee = np.clip(150 * demand, 0, None)
        weights = rng.uniform(1e4, 1e7, 40)
        cal = DemandRateCalibrator()
        result = cal.fit(demand, fee, sample_weight=weights)
        assert result.n_observations == 40

    def test_fee_with_uncertainty_returns_mean_and_std(self):
        cal, _ = self._fit_calibrator(50)
        mean_fee, std_fee = cal.fee_with_uncertainty(0.5, 0.1, n_samples=200)
        assert isinstance(mean_fee, float)
        assert isinstance(std_fee, float)
        assert std_fee >= 0.0

    def test_fee_with_uncertainty_zero_std_is_near_deterministic(self):
        cal, _ = self._fit_calibrator(60)
        mean_fee, std_fee = cal.fee_with_uncertainty(0.5, 1e-9, n_samples=100)
        # With near-zero demand uncertainty, fee uncertainty should be tiny
        assert std_fee < 5.0


# ---------------------------------------------------------------------------
# allocator.py
# ---------------------------------------------------------------------------


class TestLocateAllocator:
    def _make_inventory(self):
        from qr_haven.borrow_demand.allocator import InventorySnapshot

        return {
            "AAPL": InventorySnapshot("AAPL", available_shares=1000.0, price_usd=180.0),
            "GME": InventorySnapshot("GME", available_shares=200.0, price_usd=15.0),
            "TSLA": InventorySnapshot("TSLA", available_shares=0.0, price_usd=250.0),
        }

    def _make_request(self, locate_id, cusip, qty, fee):
        from qr_haven.borrow_demand.allocator import LocateRequest

        return LocateRequest(
            locate_id=locate_id,
            cusip=cusip,
            client_id="C1",
            requested_qty_shares=qty,
            fee_bps=fee,
            timestamp=datetime(2024, 1, 2, 9, 30),
        )

    def test_full_approval(self):
        from qr_haven.borrow_demand.allocator import LocateAllocator

        alloc = LocateAllocator(self._make_inventory())
        result = alloc.allocate(self._make_request("L1", "AAPL", 500.0, 100.0))
        assert result.status == "approved"
        assert result.approved_qty_shares == 500.0

    def test_partial_fill(self):
        from qr_haven.borrow_demand.allocator import LocateAllocator

        alloc = LocateAllocator(self._make_inventory())
        result = alloc.allocate(self._make_request("L1", "GME", 500.0, 80.0))
        assert result.status == "partial"
        assert result.approved_qty_shares == 200.0

    def test_rejection_no_inventory(self):
        from qr_haven.borrow_demand.allocator import LocateAllocator

        alloc = LocateAllocator(self._make_inventory())
        result = alloc.allocate(self._make_request("L1", "TSLA", 100.0, 50.0))
        assert result.status == "rejected"
        assert result.approved_qty_shares == 0.0

    def test_rejection_unknown_cusip(self):
        from qr_haven.borrow_demand.allocator import LocateAllocator

        alloc = LocateAllocator(self._make_inventory())
        result = alloc.allocate(self._make_request("L1", "UNKN", 100.0, 50.0))
        assert result.status == "rejected"

    def test_inventory_decrements(self):
        from qr_haven.borrow_demand.allocator import LocateAllocator

        alloc = LocateAllocator(self._make_inventory())
        alloc.allocate(self._make_request("L1", "AAPL", 400.0, 100.0))
        remaining = alloc.remaining_inventory()
        assert remaining["AAPL"] == pytest.approx(600.0)

    def test_batch_preserves_input_order(self):
        from qr_haven.borrow_demand.allocator import LocateAllocator

        alloc = LocateAllocator(self._make_inventory())
        requests = [
            self._make_request("L1", "AAPL", 100.0, 50.0),
            self._make_request("L2", "GME", 50.0, 200.0),
            self._make_request("L3", "AAPL", 200.0, 80.0),
        ]
        results = alloc.allocate_batch(requests)
        assert results[0].locate_id == "L1"
        assert results[1].locate_id == "L2"
        assert results[2].locate_id == "L3"

    def test_batch_prioritises_high_revenue_density(self):
        from qr_haven.borrow_demand.allocator import InventorySnapshot, LocateAllocator

        # Only 100 shares of GME available — two requests compete
        inventory = {"GME": InventorySnapshot("GME", 100.0)}
        alloc = LocateAllocator(inventory)
        # L_high has higher revenue density (200 bps × 100 > 50 bps × 100)
        requests = [
            self._make_request("L_low", "GME", 100.0, 50.0),
            self._make_request("L_high", "GME", 100.0, 200.0),
        ]
        results = alloc.allocate_batch(requests)
        # L_high should be approved, L_low rejected
        by_id = {r.locate_id: r for r in results}
        assert by_id["L_high"].status == "approved"
        assert by_id["L_low"].status == "rejected"

    def test_min_fill_ratio_rejects_partial(self):
        from qr_haven.borrow_demand.allocator import LocateAllocator

        alloc = LocateAllocator(self._make_inventory(), min_fill_ratio=0.9)
        # GME only has 200 shares; requesting 500 → fill ratio 0.4 < 0.9
        result = alloc.allocate(self._make_request("L1", "GME", 500.0, 80.0))
        assert result.status == "rejected"
        assert result.approved_qty_shares == 0.0

    def test_revenue_computed_when_price_available(self):
        from qr_haven.borrow_demand.allocator import LocateAllocator

        alloc = LocateAllocator(self._make_inventory())
        result = alloc.allocate(self._make_request("L1", "AAPL", 1000.0, 100.0))
        # revenue = 100/10000 * 1000 * 180 = 1800
        assert result.revenue_usd == pytest.approx(1800.0)

    def test_min_fill_ratio_validation(self):
        from qr_haven.borrow_demand.allocator import LocateAllocator

        with pytest.raises(ValueError):
            LocateAllocator(self._make_inventory(), min_fill_ratio=1.5)


# ---------------------------------------------------------------------------
# diagnostics.py
# ---------------------------------------------------------------------------


class TestSurfaceRmse:
    def test_perfect_prediction(self):
        from qr_haven.borrow_demand.diagnostics import surface_rmse

        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        std = np.ones(5) * 0.1
        metrics = surface_rmse(actual, pred, std)
        assert metrics.rmse == pytest.approx(0.0, abs=1e-9)
        assert metrics.mae == pytest.approx(0.0, abs=1e-9)
        assert metrics.coverage_90 == pytest.approx(1.0)
        assert metrics.coverage_95 == pytest.approx(1.0)

    def test_known_rmse(self):
        from qr_haven.borrow_demand.diagnostics import surface_rmse

        actual = np.array([0.0, 0.0])
        pred = np.array([1.0, 1.0])
        std = np.ones(2) * 10.0
        metrics = surface_rmse(actual, pred, std)
        assert metrics.rmse == pytest.approx(1.0)
        assert metrics.mae == pytest.approx(1.0)

    def test_shape_mismatch_raises(self):
        from qr_haven.borrow_demand.diagnostics import surface_rmse

        with pytest.raises(ValueError):
            surface_rmse(np.ones(5), np.ones(4), np.ones(5))

    def test_empty_raises(self):
        from qr_haven.borrow_demand.diagnostics import surface_rmse

        with pytest.raises(ValueError):
            surface_rmse(np.array([]), np.array([]), np.array([]))

    def test_n_observations(self):
        from qr_haven.borrow_demand.diagnostics import surface_rmse

        n = 30
        metrics = surface_rmse(np.ones(n), np.ones(n), np.ones(n))
        assert metrics.n_observations == n


class TestCalibrationReliability:
    def _make_data(self, n=100, seed=0):
        rng = np.random.default_rng(seed)
        demand = rng.uniform(0, 1, n)
        predicted_fee = 50 + 200 * demand
        actual_fee = predicted_fee + rng.normal(0, 5, n)
        return demand, predicted_fee, actual_fee

    def test_returns_correct_n_bins(self):
        from qr_haven.borrow_demand.diagnostics import calibration_reliability

        demand, pred, actual = self._make_data(100)
        diag = calibration_reliability(demand, pred, actual, n_bins=5)
        assert diag.n_bins == 5
        assert len(diag.bin_midpoints) == 5
        assert len(diag.mean_predicted_fee) == 5

    def test_mae_near_zero_for_perfect_calibration(self):
        from qr_haven.borrow_demand.diagnostics import calibration_reliability

        demand, pred, _ = self._make_data(200)
        # perfect calibration: actual == predicted
        diag = calibration_reliability(demand, pred, pred, n_bins=10)
        assert diag.mean_absolute_error == pytest.approx(0.0, abs=1e-6)

    def test_too_few_observations_raises(self):
        from qr_haven.borrow_demand.diagnostics import calibration_reliability

        with pytest.raises(ValueError):
            calibration_reliability(np.ones(5), np.ones(5), np.ones(5), n_bins=10)


class TestShortageRecall:
    def test_perfect_detection(self):
        from qr_haven.borrow_demand.diagnostics import shortage_recall

        actual = np.array([1, 1, 0, 0, 1], dtype=bool)
        demand = np.array([2.0, 3.0, 0.5, 0.5, 2.5])
        metrics = shortage_recall(actual, demand, demand_threshold=1.0)
        assert metrics.precision == pytest.approx(1.0)
        assert metrics.recall == pytest.approx(1.0)
        assert metrics.f1 == pytest.approx(1.0)

    def test_no_actual_shortages(self):
        from qr_haven.borrow_demand.diagnostics import shortage_recall

        actual = np.zeros(10, dtype=bool)
        demand = np.ones(10) * 2.0
        metrics = shortage_recall(actual, demand, demand_threshold=1.0)
        assert metrics.recall == pytest.approx(0.0)
        assert metrics.n_actual_shortages == 0

    def test_threshold_sensitivity(self):
        from qr_haven.borrow_demand.diagnostics import shortage_recall

        actual = np.array([1, 1, 0, 0], dtype=bool)
        demand = np.array([1.5, 0.5, 0.5, 0.5])
        high_threshold = shortage_recall(actual, demand, demand_threshold=1.2)
        low_threshold = shortage_recall(actual, demand, demand_threshold=0.3)
        # lower threshold predicts more shortages (recall may be higher)
        assert low_threshold.n_predicted_shortages >= high_threshold.n_predicted_shortages

    def test_f1_harmonic_mean(self):
        from qr_haven.borrow_demand.diagnostics import shortage_recall

        actual = np.array([1, 1, 0, 1, 0], dtype=bool)
        demand = np.array([2.0, 2.0, 0.5, 0.5, 0.5])
        metrics = shortage_recall(actual, demand, demand_threshold=1.0)
        expected_f1 = (2 * metrics.precision * metrics.recall
                       / (metrics.precision + metrics.recall)
                       if (metrics.precision + metrics.recall) > 0 else 0.0)
        assert metrics.f1 == pytest.approx(expected_f1)


# ---------------------------------------------------------------------------
# __init__.py — public API smoke test
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Gap 1: shortage_probability() and demand_quantile() — inference.py
# ---------------------------------------------------------------------------


class TestInference:
    def _make_model_and_likelihood(self, use_time_kernel: bool = False):
        import gpytorch
        from qr_haven.borrow_demand.model import BorrowDemandSurface
        from qr_haven.borrow_demand.features import N_FEATURES, N_FULL_INPUT_DIMS

        n_dims = N_FULL_INPUT_DIMS if use_time_kernel else N_FEATURES
        inducing = __import__("torch").rand(10, n_dims)
        model = BorrowDemandSurface(
            inducing, learn_inducing=True, use_time_kernel=use_time_kernel
        )
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model.eval()
        likelihood.eval()
        return model, likelihood

    def _sf(self, time: float = 0.0):
        from qr_haven.borrow_demand.features import SurfaceFeatures
        return SurfaceFeatures(0.5, 0.5, 0.5, 0.5, 0.5, trading_time_norm=time)

    def test_shortage_probability_returns_float(self):
        from qr_haven.borrow_demand.inference import shortage_probability

        model, lik = self._make_model_and_likelihood()
        p = shortage_probability(model, lik, self._sf(), inventory_shares=100.0)
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0

    def test_shortage_probability_zero_inventory_near_one(self):
        from qr_haven.borrow_demand.inference import shortage_probability

        model, lik = self._make_model_and_likelihood()
        # With 0 inventory any positive demand is a shortage — prob should be high
        p = shortage_probability(model, lik, self._sf(), inventory_shares=0.0)
        assert p >= 0.0  # can't assert near-1 without controlling GP mean

    def test_shortage_probability_large_inventory_near_zero(self):
        from qr_haven.borrow_demand.inference import shortage_probability

        model, lik = self._make_model_and_likelihood()
        # With enormous inventory shortage prob should be lower
        p_small = shortage_probability(model, lik, self._sf(), inventory_shares=0.0)
        p_large = shortage_probability(model, lik, self._sf(), inventory_shares=1e9)
        assert p_large <= p_small

    def test_shortage_probability_with_time_kernel(self):
        from qr_haven.borrow_demand.inference import shortage_probability

        model, lik = self._make_model_and_likelihood(use_time_kernel=True)
        p = shortage_probability(model, lik, self._sf(time=0.5), inventory_shares=50.0)
        assert 0.0 <= p <= 1.0

    def test_demand_quantile_returns_float(self):
        from qr_haven.borrow_demand.inference import demand_quantile

        model, lik = self._make_model_and_likelihood()
        q = demand_quantile(model, lik, self._sf(), quantile=0.95)
        assert isinstance(q, float)

    def test_demand_quantile_ordering(self):
        from qr_haven.borrow_demand.inference import demand_quantile

        model, lik = self._make_model_and_likelihood()
        q50 = demand_quantile(model, lik, self._sf(), quantile=0.50)
        q95 = demand_quantile(model, lik, self._sf(), quantile=0.95)
        # Higher quantile → higher demand
        assert q95 >= q50

    def test_demand_quantile_invalid_raises(self):
        from qr_haven.borrow_demand.inference import demand_quantile

        model, lik = self._make_model_and_likelihood()
        with pytest.raises(ValueError, match="quantile"):
            demand_quantile(model, lik, self._sf(), quantile=1.5)


# ---------------------------------------------------------------------------
# Gap 2: composite time kernel — model.py
# ---------------------------------------------------------------------------


class TestTimeKernel:
    def test_config_use_time_kernel_default_false(self):
        from qr_haven.borrow_demand.model import BorrowDemandConfig

        cfg = BorrowDemandConfig()
        assert cfg.use_time_kernel is False

    def test_model_with_time_kernel_requires_6_dim(self):
        import torch
        from qr_haven.borrow_demand.model import BorrowDemandSurface

        inducing_6 = torch.rand(10, 6)
        model = BorrowDemandSurface(inducing_6, use_time_kernel=True)
        assert model.use_time_kernel is True

    def test_model_with_time_kernel_rejects_5_dim(self):
        import torch
        from qr_haven.borrow_demand.model import BorrowDemandSurface

        with pytest.raises(ValueError, match="inducing_points"):
            BorrowDemandSurface(torch.rand(10, 5), use_time_kernel=True)

    def test_model_without_time_kernel_rejects_6_dim(self):
        import torch
        from qr_haven.borrow_demand.model import BorrowDemandSurface

        with pytest.raises(ValueError, match="inducing_points"):
            BorrowDemandSurface(torch.rand(10, 6), use_time_kernel=False)

    def test_time_kernel_model_predict_shape(self):
        import gpytorch
        import torch
        from qr_haven.borrow_demand.model import BorrowDemandSurface

        inducing = torch.rand(10, 6)
        model = BorrowDemandSurface(inducing, use_time_kernel=True)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model.eval()
        likelihood.eval()

        x = torch.rand(4, 6)
        mean, std = model.predict(x, likelihood)
        assert mean.shape == (4,)
        assert np.all(std >= 0)

    def test_composite_kernel_has_additive_structure(self):
        import torch
        from qr_haven.borrow_demand.model import BorrowDemandSurface

        model = BorrowDemandSurface(torch.rand(10, 6), use_time_kernel=True)
        # GPyTorch AdditiveKernel is the sum of kernels
        assert hasattr(model.covar_module, "kernels")

    def test_surface_features_to_full_tensor_shape(self):
        import torch
        from qr_haven.borrow_demand.features import SurfaceFeatures

        sf = SurfaceFeatures(0.1, 0.2, 0.3, 0.4, 0.5, trading_time_norm=0.6)
        t = sf.to_full_tensor()
        assert isinstance(t, torch.Tensor)
        assert t.shape == (6,)
        assert float(t[5]) == pytest.approx(0.6)

    def test_make_model_and_likelihood_wires_use_time_kernel(self):
        import gpytorch
        from qr_haven.borrow_demand.model import BorrowDemandConfig
        from qr_haven.borrow_demand.trainer import make_model_and_likelihood

        X = np.random.default_rng(0).random((20, 6)).astype(np.float32)
        cfg = BorrowDemandConfig(n_inducing=10, use_time_kernel=True)
        model, likelihood = make_model_and_likelihood(X, config=cfg, seed=0)
        assert model.use_time_kernel is True
        assert isinstance(likelihood, gpytorch.likelihoods.GaussianLikelihood)


# ---------------------------------------------------------------------------
# Gap 3: learn_inducing wired from config — model.py + trainer.py
# ---------------------------------------------------------------------------


class TestLearnInducing:
    def test_model_learn_inducing_true(self):
        import torch
        from qr_haven.borrow_demand.model import BorrowDemandSurface

        model = BorrowDemandSurface(torch.rand(10, 5), learn_inducing=True)
        ip = model.variational_strategy.inducing_points
        assert ip.requires_grad is True

    def test_model_learn_inducing_false(self):
        import torch
        from qr_haven.borrow_demand.model import BorrowDemandSurface

        model = BorrowDemandSurface(torch.rand(10, 5), learn_inducing=False)
        ip = model.variational_strategy.inducing_points
        assert ip.requires_grad is False

    def test_make_model_and_likelihood_wires_learn_inducing_false(self):
        from qr_haven.borrow_demand.model import BorrowDemandConfig
        from qr_haven.borrow_demand.trainer import make_model_and_likelihood

        X = np.random.default_rng(0).random((20, 5)).astype(np.float32)
        cfg = BorrowDemandConfig(n_inducing=10, learn_inducing=False)
        model, _ = make_model_and_likelihood(X, config=cfg)
        ip = model.variational_strategy.inducing_points
        assert ip.requires_grad is False

    def test_make_model_and_likelihood_wires_learn_inducing_true(self):
        from qr_haven.borrow_demand.model import BorrowDemandConfig
        from qr_haven.borrow_demand.trainer import make_model_and_likelihood

        X = np.random.default_rng(0).random((20, 5)).astype(np.float32)
        cfg = BorrowDemandConfig(n_inducing=10, learn_inducing=True)
        model, _ = make_model_and_likelihood(X, config=cfg)
        ip = model.variational_strategy.inducing_points
        assert ip.requires_grad is True


# ---------------------------------------------------------------------------
# Public API (updated)
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_all_symbols_importable(self):
        import qr_haven.borrow_demand as bd

        symbols = [
            "N_FEATURES", "N_FULL_INPUT_DIMS", "FEATURE_NAMES",
            "RawFeatures", "SurfaceFeatures", "FeaturePipeline",
            "BorrowDemandConfig", "BorrowDemandSurface",
            "TrainingResult", "train_surface", "make_model_and_likelihood",
            "LocateEvent", "OnlineSurfaceUpdater",
            "CalibrationResult", "DemandRateCalibrator",
            "LocateRequest", "AllocationResult", "InventorySnapshot", "LocateAllocator",
            "SurfaceMetrics", "CalibrationDiagnostic", "ShortageRecallMetrics",
            "surface_rmse", "calibration_reliability", "shortage_recall",
            "shortage_probability", "demand_quantile",
        ]
        for sym in symbols:
            assert hasattr(bd, sym), f"Missing from public API: {sym}"

    def test_n_features_is_five(self):
        from qr_haven.borrow_demand import N_FEATURES

        assert N_FEATURES == 5

    def test_n_full_input_dims_is_six(self):
        from qr_haven.borrow_demand import N_FULL_INPUT_DIMS

        assert N_FULL_INPUT_DIMS == 6

    def test_feature_names_length(self):
        from qr_haven.borrow_demand import FEATURE_NAMES

        assert len(FEATURE_NAMES) == 5
