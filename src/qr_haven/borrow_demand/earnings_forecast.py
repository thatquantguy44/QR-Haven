"""Earnings-window borrow demand forecast (Extension 1).

Two complementary models predict the borrow-fee arc around earnings:

  A  PanelRegressionForecaster — within-estimator OLS with CUSIP fixed
     effects and polynomial τ terms.  Fast; no ML; interpretable baseline.

  B  LSTMFeeForecaster — 2-layer stacked LSTM (pure numpy, no PyTorch) that
     maps a 60-day pre-earnings feature sequence to the 20-day post-earnings
     fee trajectory.  Architecture: hidden_size=64, 2 layers, dropout=0.2.

  EarningsWindowForecaster wraps both and returns EarningsForecastResult.

Reference: spec § Extension Model 1 — Earnings-Window Borrow Demand Forecast.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


@dataclass
class EarningsRecord:
    """Single confirmed earnings announcement for one CUSIP."""

    cusip: str
    earnings_date: date
    eps_actual: float = float("nan")
    eps_consensus: float = float("nan")
    eps_surprise_pct: float = float("nan")   # (actual − consensus) / |consensus|
    sector: str = ""


@dataclass
class EarningsForecastResult:
    """Fee trajectory forecast for one CUSIP around one earnings event."""

    cusip: str
    earnings_date: date
    as_of_date: date
    current_tau: int                # calendar days: as_of_date − earnings_date
    predicted_fee_bps: np.ndarray   # (forecast_days,) at τ+1 … τ+forecast_days
    predicted_fee_bps_T10: float    # fee predicted 10 days from as_of_date
    expected_peak_fee: float        # max(predicted_fee_bps)
    expected_peak_day: int          # days from as_of_date where peak occurs (1-based)
    term_vs_overnight_signal: float  # >0 → term borrow cheaper than rolling overnight
    model_used: str                 # "panel_regression" | "lstm"


# ---------------------------------------------------------------------------
# Internal 2-layer LSTM primitives
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _step_fwd(
    x: np.ndarray,
    h: np.ndarray,
    c: np.ndarray,
    Wx: np.ndarray,
    Wh: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Single LSTM cell forward step. Returns (h_new, c_new, cache)."""
    H = h.shape[0]
    z = Wx @ x + Wh @ h + b
    i_g = _sigmoid(z[:H])
    f_g = _sigmoid(z[H : 2 * H])
    g_g = np.tanh(z[2 * H : 3 * H])
    o_g = _sigmoid(z[3 * H :])
    c_new = f_g * c + i_g * g_g
    h_new = o_g * np.tanh(c_new)
    cache = {
        "x": x, "h_prev": h, "c_prev": c,
        "i": i_g, "f": f_g, "g": g_g, "o": o_g, "c": c_new, "h": h_new,
    }
    return h_new, c_new, cache


def _step_bwd(
    dh: np.ndarray,
    dc_next: np.ndarray,
    cache: dict,
    Wx: np.ndarray,
    Wh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Backward through one LSTM cell.

    Returns (dx, dh_prev, dc_prev, dWx, dWh, db).
    """
    i_g, f_g, g_g, o_g = cache["i"], cache["f"], cache["g"], cache["o"]
    c, c_prev = cache["c"], cache["c_prev"]
    h_prev, x = cache["h_prev"], cache["x"]
    tanh_c = np.tanh(c)

    dc = dh * o_g * (1.0 - tanh_c ** 2) + dc_next
    do = dh * tanh_c
    di = dc * g_g
    df = dc * c_prev
    dg = dc * i_g
    dc_prev = dc * f_g

    dz = np.concatenate([
        di * i_g * (1.0 - i_g),
        df * f_g * (1.0 - f_g),
        dg * (1.0 - g_g ** 2),
        do * o_g * (1.0 - o_g),
    ])
    return Wx.T @ dz, Wh.T @ dz, dc_prev, np.outer(dz, x), np.outer(dz, h_prev), dz


# ---------------------------------------------------------------------------
# 2-layer LSTM parameter and optimizer containers
# ---------------------------------------------------------------------------


class _Params2L:
    """Weight matrices: 2-layer LSTM + multi-output linear head."""

    __slots__ = ("Wx1", "Wh1", "b1", "Wx2", "Wh2", "b2", "Wy", "by")

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        s1 = np.sqrt(2.0 / (input_size + hidden_size))
        s2 = np.sqrt(2.0 / (2 * hidden_size))
        s_out = np.sqrt(2.0 / hidden_size)
        self.Wx1 = rng.normal(0.0, s1, (4 * hidden_size, input_size))
        self.Wh1 = rng.normal(0.0, s1, (4 * hidden_size, hidden_size))
        self.b1 = np.zeros(4 * hidden_size)
        self.b1[hidden_size : 2 * hidden_size] = 1.0   # forget-gate bias
        self.Wx2 = rng.normal(0.0, s2, (4 * hidden_size, hidden_size))
        self.Wh2 = rng.normal(0.0, s2, (4 * hidden_size, hidden_size))
        self.b2 = np.zeros(4 * hidden_size)
        self.b2[hidden_size : 2 * hidden_size] = 1.0
        self.Wy = rng.normal(0.0, s_out, (output_size, hidden_size))
        self.by = np.zeros(output_size)


class _Adam2L:
    """Adam optimizer first/second moment state for _Params2L."""

    _KEYS = ("Wx1", "Wh1", "b1", "Wx2", "Wh2", "b2", "Wy", "by")

    def __init__(self, params: _Params2L) -> None:
        self.m = {k: np.zeros_like(getattr(params, k)) for k in self._KEYS}
        self.v = {k: np.zeros_like(getattr(params, k)) for k in self._KEYS}
        self.t: int = 0


def _adam2l_step(
    params: _Params2L,
    grads: dict[str, np.ndarray],
    adam: _Adam2L,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> None:
    adam.t += 1
    bc1 = 1.0 - beta1 ** adam.t
    bc2 = 1.0 - beta2 ** adam.t
    for k in adam.m:
        g = grads[k]
        m, v = adam.m[k], adam.v[k]
        m[:] = beta1 * m + (1.0 - beta1) * g
        v[:] = beta2 * v + (1.0 - beta2) * g ** 2
        getattr(params, k)[:] -= lr * (m / bc1) / (np.sqrt(v / bc2) + eps)


# ---------------------------------------------------------------------------
# 2-layer LSTM forward / backward
# ---------------------------------------------------------------------------


def _forward2L(
    X: np.ndarray,
    params: _Params2L,
    dropout_rate: float,
    rng: np.random.Generator | None,
    training: bool,
) -> tuple[np.ndarray, list, np.ndarray, list, np.ndarray | None]:
    """2-layer LSTM forward pass. Returns (y_hat, caches1, h1_dropped, caches2, mask)."""
    T, _ = X.shape
    H = params.Wh1.shape[1]

    h1, c1 = np.zeros(H), np.zeros(H)
    h1_seq = np.empty((T, H))
    caches1: list[dict] = []
    for t in range(T):
        h1, c1, cache = _step_fwd(X[t], h1, c1, params.Wx1, params.Wh1, params.b1)
        h1_seq[t] = h1
        caches1.append(cache)

    # Inverted dropout between layers
    if dropout_rate > 0.0 and training and rng is not None:
        mask: np.ndarray | None = (
            (rng.random((T, H)) >= dropout_rate).astype(np.float64)
            / (1.0 - dropout_rate)
        )
        h1_dropped = h1_seq * mask
    else:
        mask = None
        h1_dropped = h1_seq

    h2, c2 = np.zeros(H), np.zeros(H)
    caches2: list[dict] = []
    for t in range(T):
        h2, c2, cache = _step_fwd(h1_dropped[t], h2, c2, params.Wx2, params.Wh2, params.b2)
        caches2.append(cache)

    y_hat = params.Wy @ h2 + params.by   # (output_size,)
    return y_hat, caches1, h1_dropped, caches2, mask


def _backward2L(
    dy_hat: np.ndarray,
    caches1: list,
    h1_dropped: np.ndarray,
    caches2: list,
    mask: np.ndarray | None,
    params: _Params2L,
) -> dict[str, np.ndarray]:
    """BPTT through 2-layer LSTM. Returns gradient dict."""
    T = len(caches1)
    H = params.Wh1.shape[1]

    # --- output head ---
    h2_final = caches2[-1]["h"]
    dWy = np.outer(dy_hat, h2_final)
    dby = dy_hat.copy()
    dh2 = params.Wy.T @ dy_hat   # (H,) gradient into final h2

    # --- layer 2 BPTT ---
    dWx2 = np.zeros_like(params.Wx2)
    dWh2 = np.zeros_like(params.Wh2)
    db2 = np.zeros_like(params.b2)
    dh1_dropped = np.zeros((T, H))
    dc2 = np.zeros(H)

    for t in reversed(range(T)):
        dx2, dh2, dc2, dWx2_t, dWh2_t, db2_t = _step_bwd(
            dh2, dc2, caches2[t], params.Wx2, params.Wh2
        )
        dh1_dropped[t] = dx2   # gradient w.r.t. input to layer 2 = h1_dropped[t]
        dWx2 += dWx2_t
        dWh2 += dWh2_t
        db2 += db2_t

    # --- dropout backward ---
    dh1_seq = dh1_dropped * mask if mask is not None else dh1_dropped

    # --- layer 1 BPTT (gradients from both layer-2 inputs and BPTT through time) ---
    dWx1 = np.zeros_like(params.Wx1)
    dWh1 = np.zeros_like(params.Wh1)
    db1 = np.zeros_like(params.b1)
    dh1 = np.zeros(H)
    dc1 = np.zeros(H)

    for t in reversed(range(T)):
        _, dh1, dc1, dWx1_t, dWh1_t, db1_t = _step_bwd(
            dh1 + dh1_seq[t], dc1, caches1[t], params.Wx1, params.Wh1
        )
        dWx1 += dWx1_t
        dWh1 += dWh1_t
        db1 += db1_t

    return {
        "Wx1": dWx1, "Wh1": dWh1, "b1": db1,
        "Wx2": dWx2, "Wh2": dWh2, "b2": db2,
        "Wy": dWy, "by": dby,
    }


# ---------------------------------------------------------------------------
# EarningsPanelBuilder
# ---------------------------------------------------------------------------


_FEATURE_COLS = ("fee_bps", "si_ratio", "utilization", "volume_ratio")


class EarningsPanelBuilder:
    """Build τ-aligned panels for training and inference.

    τ = calendar_date − earnings_date (negative = pre-earnings).

    Parameters
    ----------
    lookback_days : int
        Number of pre-earnings days used as LSTM input window (τ ∈ [−lookback, 0]).
    forecast_days : int
        Number of post-earnings days to predict (τ ∈ [+1, +forecast_days]).
    """

    def __init__(self, lookback_days: int = 60, forecast_days: int = 20) -> None:
        if lookback_days < 1 or forecast_days < 1:
            raise ValueError("lookback_days and forecast_days must be ≥ 1")
        self.lookback_days = lookback_days
        self.forecast_days = forecast_days

    # ---- Panel DataFrame (used by PanelRegressionForecaster) ----

    def build_panel(
        self,
        fee_df: pd.DataFrame,
        earnings_records: list[EarningsRecord],
    ) -> pd.DataFrame:
        """Build a flat τ-indexed panel from daily fee data and earnings records.

        fee_df must have columns: cusip (str), trade_date (date/str),
        fee_bps (float).  Optionally: si_ratio, utilization, volume_ratio.

        Returns DataFrame with columns: cusip, tau, fee_bps, [si_ratio,
        utilization, volume_ratio], earnings_date.  Only rows within the
        window [−lookback_days, +forecast_days] are included.
        """
        df = fee_df.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date

        rows = []
        for er in earnings_records:
            sub = df[df["cusip"] == er.cusip].copy()
            if sub.empty:
                continue
            sub["tau"] = sub["trade_date"].apply(
                lambda d: (d - er.earnings_date).days
            )
            window = sub[
                (sub["tau"] >= -self.lookback_days) & (sub["tau"] <= self.forecast_days)
            ].copy()
            if window.empty:
                continue
            window["earnings_date"] = er.earnings_date
            rows.append(window)

        if not rows:
            cols = ["cusip", "tau", "fee_bps", "earnings_date"]
            for c in ("si_ratio", "utilization", "volume_ratio"):
                if c in fee_df.columns:
                    cols.append(c)
            return pd.DataFrame(columns=cols)

        panel = pd.concat(rows, ignore_index=True)
        keep = ["cusip", "tau", "fee_bps", "earnings_date"]
        for c in ("si_ratio", "utilization", "volume_ratio"):
            if c in panel.columns:
                keep.append(c)
        return panel[keep]

    # ---- LSTM sequence tensors ----

    def build_sequences(
        self,
        fee_df: pd.DataFrame,
        earnings_records: list[EarningsRecord],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build (X, y) arrays for LSTM training.

        X shape: (N, lookback_days, n_features)  where features ∈ {fee_bps,
        si_ratio, utilization, volume_ratio} (NaN-filled columns omitted).
        y shape: (N, forecast_days).

        Pairs where any target day is missing are dropped.  Lookback windows
        shorter than lookback_days are left-padded with the earliest available
        value (or zeros if entirely missing).
        """
        df = fee_df.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date

        avail_feats = [c for c in _FEATURE_COLS if c in df.columns]
        if "fee_bps" not in avail_feats:
            raise ValueError("fee_df must contain 'fee_bps'")

        Xs, ys = [], []

        for er in earnings_records:
            sub = df[df["cusip"] == er.cusip].copy()
            if sub.empty:
                continue

            sub["tau"] = sub["trade_date"].apply(
                lambda d: (d - er.earnings_date).days
            )
            sub = sub.sort_values("tau")

            # --- input window: tau in [-lookback_days, 0] ---
            inp = sub[(sub["tau"] >= -self.lookback_days) & (sub["tau"] <= 0)]
            # --- target window: tau in [1, forecast_days] ---
            tgt = sub[(sub["tau"] >= 1) & (sub["tau"] <= self.forecast_days)]

            if len(tgt) < self.forecast_days:
                continue  # skip if any target day is missing

            # Build dense tau-indexed arrays for input
            tau_range = list(range(-self.lookback_days, 1))
            inp_indexed = inp.set_index("tau")[avail_feats].reindex(tau_range)
            # Forward-fill then backward-fill within the window, then zero-fill
            inp_indexed = inp_indexed.ffill().bfill().fillna(0.0)
            X_seq = inp_indexed.to_numpy(dtype=np.float32)   # (lookback+1, F); drop τ=0

            # Drop the τ=0 row (present day) — only use τ ∈ [-lookback, -1]
            X_seq = X_seq[:-1]   # (lookback_days, F)

            # Target: fee_bps at tau ∈ [1, forecast_days]
            tgt_tau = (
                tgt.set_index("tau")["fee_bps"]
                .reindex(range(1, self.forecast_days + 1))
                .to_numpy(dtype=np.float32)
            )
            if np.any(np.isnan(tgt_tau)):
                continue

            Xs.append(X_seq)
            ys.append(tgt_tau)

        if not Xs:
            n_f = len(avail_feats)
            return (
                np.empty((0, self.lookback_days, n_f), dtype=np.float32),
                np.empty((0, self.forecast_days), dtype=np.float32),
            )

        return np.stack(Xs).astype(np.float32), np.stack(ys).astype(np.float32)

    def build_input_sequence(
        self,
        fee_df: pd.DataFrame,
        cusip: str,
        earnings_date: date,
        as_of_date: date,
    ) -> np.ndarray:
        """Build a single input sequence for LSTM inference.

        Returns array of shape (lookback_days, n_features) aligned to the 60
        days ending one day before as_of_date (τ relative to earnings_date).
        Gaps are forward/backward-filled; leading NaNs are zero-padded.
        """
        df = fee_df.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        sub = df[df["cusip"] == cusip].copy()

        avail_feats = [c for c in _FEATURE_COLS if c in sub.columns]
        if "fee_bps" not in avail_feats:
            raise ValueError("fee_df must contain 'fee_bps'")

        current_tau = (as_of_date - earnings_date).days
        start_tau = current_tau - self.lookback_days

        if sub.empty:
            return np.zeros((self.lookback_days, len(avail_feats)), dtype=np.float32)

        sub["tau"] = sub["trade_date"].apply(lambda d: (d - earnings_date).days)
        sub = sub[(sub["tau"] >= start_tau) & (sub["tau"] < current_tau)]

        tau_range = list(range(start_tau, current_tau))
        indexed = sub.set_index("tau")[avail_feats].reindex(tau_range)
        indexed = indexed.ffill().bfill().fillna(0.0)
        return indexed.to_numpy(dtype=np.float32)


# ---------------------------------------------------------------------------
# PanelRegressionForecaster (model A)
# ---------------------------------------------------------------------------


class PanelRegressionForecaster:
    """Within-estimator OLS with CUSIP fixed effects.

    Model: fee_bps(i, τ) = α_i + β₁·τ + β₂·τ² + β₃·si_ratio + β₄·utilization + ε

    Fixed effects absorbed by within-estimation (subtract CUSIP means before
    running OLS).  For CUSIPs not seen during training, α_i is set to the
    cross-sectional mean of training intercepts.
    """

    def __init__(self) -> None:
        self._beta: np.ndarray | None = None             # (4,) coefficients
        self._cusip_alpha: dict[str, float] = {}         # per-CUSIP intercepts
        self._global_alpha: float = 0.0
        self._fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, panel_df: pd.DataFrame) -> PanelRegressionForecaster:
        """Fit on a τ-aligned panel produced by EarningsPanelBuilder.build_panel().

        Required columns: cusip, tau, fee_bps.
        Optional columns: si_ratio, utilization (filled with 0 if missing).
        """
        df = panel_df.copy()
        for col in ("si_ratio", "utilization"):
            if col not in df.columns:
                df[col] = 0.0
        df = df.dropna(subset=["tau", "fee_bps"])

        if len(df) < 4:
            raise ValueError("panel_df has fewer than 4 non-null rows; cannot fit")

        # Within-estimation: subtract CUSIP means
        cusip_means = df.groupby("cusip")[["fee_bps", "tau", "si_ratio", "utilization"]].transform("mean")
        fee_dem = (df["fee_bps"] - cusip_means["fee_bps"]).to_numpy()
        tau_dem = (df["tau"] - cusip_means["tau"]).to_numpy()
        tau2_dem = (df["tau"] ** 2 - cusip_means["tau"] ** 2).to_numpy()
        si_dem = (df["si_ratio"] - cusip_means["si_ratio"]).to_numpy()
        ut_dem = (df["utilization"] - cusip_means["utilization"]).to_numpy()

        X_dem = np.column_stack([tau_dem, tau2_dem, si_dem, ut_dem])
        self._beta, _, _, _ = np.linalg.lstsq(X_dem, fee_dem, rcond=None)

        # Per-CUSIP intercepts: α_i = mean(fee_i) - mean(X_i) @ β
        beta = self._beta
        for cusip, grp in df.groupby("cusip"):
            alpha_i = (
                grp["fee_bps"].mean()
                - beta[0] * grp["tau"].mean()
                - beta[1] * (grp["tau"] ** 2).mean()
                - beta[2] * grp["si_ratio"].mean()
                - beta[3] * grp["utilization"].mean()
            )
            self._cusip_alpha[cusip] = float(alpha_i)

        self._global_alpha = float(np.mean(list(self._cusip_alpha.values())))
        self._fitted = True
        return self

    def predict_fee(
        self,
        cusip: str,
        tau: int,
        si_ratio: float = 0.0,
        utilization: float = 0.0,
    ) -> float:
        """Predict fee at a single (cusip, τ) query point."""
        if not self._fitted or self._beta is None:
            raise RuntimeError("Call fit() before predict_fee()")
        alpha = self._cusip_alpha.get(cusip, self._global_alpha)
        x = np.array([tau, tau ** 2, si_ratio, utilization])
        return float(alpha + np.dot(self._beta, x))

    def predict_trajectory(
        self,
        cusip: str,
        current_tau: int,
        forecast_days: int = 20,
        si_ratio: float = 0.0,
        utilization: float = 0.0,
    ) -> np.ndarray:
        """Predict fee for τ ∈ [current_tau+1, current_tau+forecast_days].

        si_ratio and utilization are held constant at the current values across
        the forecast window (best available approximation at inference time).

        Returns array of shape (forecast_days,).
        """
        if not self._fitted or self._beta is None:
            raise RuntimeError("Call fit() before predict_trajectory()")
        taus = np.arange(current_tau + 1, current_tau + forecast_days + 1)
        alpha = self._cusip_alpha.get(cusip, self._global_alpha)
        preds = (
            alpha
            + self._beta[0] * taus
            + self._beta[1] * taus ** 2
            + self._beta[2] * si_ratio
            + self._beta[3] * utilization
        )
        return np.maximum(preds, 0.0)   # fees are non-negative


# ---------------------------------------------------------------------------
# LSTMFeeForecaster (model B)
# ---------------------------------------------------------------------------


@dataclass
class LSTMFitResult:
    """Training diagnostics from LSTMFeeForecaster.fit()."""

    n_samples: int
    n_features: int
    n_epochs_run: int
    train_loss_history: list[float]

    @property
    def final_train_loss(self) -> float:
        return self.train_loss_history[-1]


class LSTMFeeForecaster:
    """2-layer stacked LSTM for earnings-window fee trajectory prediction.

    Input:  (N, lookback_days, n_features)   where features ∈ {fee_bps, …}
    Output: (N, forecast_days)               predicted fee trajectory

    Architecture (spec default): hidden_size=64, 2 layers, dropout=0.2,
    Adam lr=5e-4, grad_clip=1.0.
    """

    def __init__(
        self,
        hidden_size: int = 64,
        dropout_rate: float = 0.2,
        learning_rate: float = 5e-4,
        n_epochs: int = 50,
        batch_size: int = 32,
        grad_clip: float = 1.0,
        seed: int = 42,
        min_train_samples: int = 20,
    ) -> None:
        if hidden_size < 1:
            raise ValueError("hidden_size must be ≥ 1")
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError("dropout_rate must be in [0, 1)")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if n_epochs < 1:
            raise ValueError("n_epochs must be ≥ 1")

        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.seed = seed
        self.min_train_samples = min_train_samples

        self._params: _Params2L | None = None
        self._n_features: int | None = None
        self._forecast_days: int | None = None
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self._fitted: bool = False
        self.fit_result: LSTMFitResult | None = None

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, X: np.ndarray, y: np.ndarray) -> LSTMFeeForecaster:
        """Train on pre-built sequences.

        Parameters
        ----------
        X : (N, T, F) float array — pre-earnings feature sequences.
        y : (N, D) float array — post-earnings fee trajectories.
        """
        if X.ndim != 3:
            raise ValueError(f"X must be 3-D (N, T, F); got {X.shape}")
        if y.ndim != 2 or len(y) != len(X):
            raise ValueError("y must be 2-D (N, D) with len(y) == len(X)")
        n_samples, T, n_features = X.shape
        forecast_days = y.shape[1]
        if n_samples < self.min_train_samples:
            raise ValueError(
                f"Need ≥ {self.min_train_samples} samples; got {n_samples}"
            )

        X = X.astype(np.float64)
        y = y.astype(np.float64)

        # Normalise inputs per-feature over (N, T)
        self._x_mean = X.mean(axis=(0, 1), keepdims=True)     # (1, 1, F)
        x_std = X.std(axis=(0, 1), keepdims=True)
        self._x_std = np.where(x_std < 1e-8, 1.0, x_std)
        X_norm = (X - self._x_mean) / self._x_std

        self._y_mean = float(y.mean())
        y_std = float(y.std())
        self._y_std = max(y_std, 1e-8)
        y_norm = (y - self._y_mean) / self._y_std

        if self._params is None or self._n_features != n_features or self._forecast_days != forecast_days:
            self._params = _Params2L(n_features, self.hidden_size, forecast_days, seed=self.seed)
            self._n_features = n_features
            self._forecast_days = forecast_days

        adam = _Adam2L(self._params)
        rng = np.random.default_rng(self.seed)
        loss_history: list[float] = []

        for _ in range(self.n_epochs):
            perm = rng.permutation(n_samples)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, self.batch_size):
                batch = perm[start : start + self.batch_size]
                Xb = X_norm[batch]   # (B, T, F)
                yb = y_norm[batch]   # (B, D)
                B = len(batch)

                grads_acc = {k: np.zeros_like(getattr(self._params, k)) for k in _Adam2L._KEYS}
                batch_loss = 0.0

                for k in range(B):
                    y_hat, caches1, h1_dropped, caches2, mask = _forward2L(
                        Xb[k], self._params, self.dropout_rate, rng, training=True
                    )
                    err = y_hat - yb[k]          # (D,)
                    batch_loss += float(np.mean(err ** 2))

                    dy_hat = (2.0 / forecast_days) * err
                    grads_k = _backward2L(dy_hat, caches1, h1_dropped, caches2, mask, self._params)
                    for key in grads_acc:
                        grads_acc[key] += grads_k[key]

                # Average + gradient clip
                for key in grads_acc:
                    grads_acc[key] /= B
                total_norm = float(np.sqrt(sum(float(np.sum(g ** 2)) for g in grads_acc.values())))
                if total_norm > self.grad_clip:
                    scale = self.grad_clip / total_norm
                    grads_acc = {k: v * scale for k, v in grads_acc.items()}

                _adam2l_step(self._params, grads_acc, adam, self.learning_rate)
                epoch_loss += batch_loss / B
                n_batches += 1

            loss_history.append(epoch_loss / max(n_batches, 1))

        self._fitted = True
        self.fit_result = LSTMFitResult(
            n_samples=n_samples,
            n_features=n_features,
            n_epochs_run=self.n_epochs,
            train_loss_history=loss_history,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict fee trajectories for N samples.

        Parameters
        ----------
        X : (N, T, F) — same feature layout as fit().

        Returns
        -------
        (N, forecast_days) predicted fee_bps (in original scale).
        """
        if not self._fitted or self._params is None:
            raise RuntimeError("Call fit() before predict()")
        if X.ndim != 3:
            raise ValueError(f"X must be 3-D (N, T, F); got {X.shape}")
        X = X.astype(np.float64)
        X_norm = (X - self._x_mean) / self._x_std
        preds = []
        for k in range(len(X_norm)):
            y_hat, _, _, _, _ = _forward2L(X_norm[k], self._params, 0.0, None, training=False)
            preds.append(y_hat * self._y_std + self._y_mean)
        out = np.stack(preds)
        return np.maximum(out, 0.0)   # fees non-negative


# ---------------------------------------------------------------------------
# EarningsWindowForecaster (top-level orchestrator)
# ---------------------------------------------------------------------------


class EarningsWindowForecaster:
    """Orchestrates both PanelRegressionForecaster and LSTMFeeForecaster.

    Parameters
    ----------
    lookback_days : int
        Pre-earnings history window (used by LSTM and panel builder).
    forecast_days : int
        Number of post-earnings days to predict.
    use_lstm : bool
        When True, LSTMFeeForecaster is used for trajectory forecasts; when
        False (or the LSTM is not fitted), falls back to panel regression.
    """

    def __init__(
        self,
        lookback_days: int = 60,
        forecast_days: int = 20,
        use_lstm: bool = True,
        hidden_size: int = 64,
        dropout_rate: float = 0.2,
        learning_rate: float = 5e-4,
        n_epochs: int = 50,
        seed: int = 42,
    ) -> None:
        self.lookback_days = lookback_days
        self.forecast_days = forecast_days
        self.use_lstm = use_lstm

        self._panel_builder = EarningsPanelBuilder(lookback_days, forecast_days)
        self._panel_reg = PanelRegressionForecaster()
        self._lstm = LSTMFeeForecaster(
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            seed=seed,
        )

    @property
    def panel_regression(self) -> PanelRegressionForecaster:
        return self._panel_reg

    @property
    def lstm_forecaster(self) -> LSTMFeeForecaster:
        return self._lstm

    def fit(
        self,
        fee_df: pd.DataFrame,
        earnings_records: list[EarningsRecord],
    ) -> EarningsWindowForecaster:
        """Fit both models on historical data.

        fee_df : DataFrame with cusip, trade_date, fee_bps and optionally
                 si_ratio, utilization, volume_ratio.
        earnings_records : list of confirmed past earnings events.
        """
        # Panel regression
        panel_df = self._panel_builder.build_panel(fee_df, earnings_records)
        if len(panel_df) >= 4:
            self._panel_reg.fit(panel_df)

        # LSTM
        if self.use_lstm:
            X, y = self._panel_builder.build_sequences(fee_df, earnings_records)
            if len(X) >= self._lstm.min_train_samples:
                self._lstm.fit(X, y)

        return self

    def forecast(
        self,
        fee_df: pd.DataFrame,
        cusip: str,
        earnings_date: date,
        as_of_date: date,
        si_ratio: float = 0.0,
        utilization: float = 0.0,
    ) -> EarningsForecastResult:
        """Produce a fee trajectory forecast for one CUSIP around one earnings event.

        Chooses LSTM if fitted and use_lstm=True, otherwise panel regression.

        Parameters
        ----------
        fee_df : DataFrame with current fee history (same schema as fit()).
        cusip : CUSIP to forecast.
        earnings_date : The upcoming (or past) earnings date.
        as_of_date : The date from which the forecast is made.
        si_ratio : Current SI ratio (used by panel regression; ignored by LSTM).
        utilization : Current utilization (used by panel regression; ignored by LSTM).
        """
        current_tau = (as_of_date - earnings_date).days

        use_lstm = self.use_lstm and self._lstm.is_fitted

        if use_lstm:
            X_seq = self._panel_builder.build_input_sequence(
                fee_df, cusip, earnings_date, as_of_date
            )
            predicted = self._lstm.predict(X_seq[np.newaxis])[0]
            model_used = "lstm"
        elif self._panel_reg.is_fitted:
            predicted = self._panel_reg.predict_trajectory(
                cusip, current_tau, self.forecast_days, si_ratio, utilization
            )
            model_used = "panel_regression"
        else:
            # Neither model is fitted yet; return zeros
            predicted = np.zeros(self.forecast_days)
            model_used = "panel_regression"

        # T+10 fee
        t10_idx = min(9, self.forecast_days - 1)
        predicted_fee_T10 = float(predicted[t10_idx])

        # Peak
        peak_idx = int(np.argmax(predicted))
        expected_peak_fee = float(predicted[peak_idx])
        expected_peak_day = peak_idx + 1   # 1-based days from as_of_date

        # Term-vs-overnight signal
        # >0 when average forecasted future fee > current fee (fees rising → lock in term)
        current_fee = self._current_fee(fee_df, cusip, as_of_date)
        term_vs_overnight = float(predicted.mean() - current_fee)

        return EarningsForecastResult(
            cusip=cusip,
            earnings_date=earnings_date,
            as_of_date=as_of_date,
            current_tau=current_tau,
            predicted_fee_bps=predicted,
            predicted_fee_bps_T10=predicted_fee_T10,
            expected_peak_fee=expected_peak_fee,
            expected_peak_day=expected_peak_day,
            term_vs_overnight_signal=term_vs_overnight,
            model_used=model_used,
        )

    def _current_fee(self, fee_df: pd.DataFrame, cusip: str, as_of_date: date) -> float:
        df = fee_df.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        sub = df[(df["cusip"] == cusip) & (df["trade_date"] <= as_of_date)]
        if sub.empty:
            return 0.0
        return float(sub.sort_values("trade_date")["fee_bps"].iloc[-1])
