"""LSTM return forecaster for cross-sectional alpha generation.

Pure numpy — no external ML framework dependency.

Architecture: a single LSTM layer with one linear output unit.
Shared weights are trained across all assets jointly (assets treated as
independent samples), so the model learns a universal mapping from factor
score sequences to forward returns.

The typical pipeline:
    1. Build per-asset factor score DataFrames (dates × features).
    2. Call LSTMReturnForecaster.generate_alpha_panel(...) which runs a
       walk-forward loop: fit on history, predict at each rebalance date,
       cross-sectionally z-score to produce a (dates × assets) alpha panel.
    3. Pass that panel to ResearchPipeline(alpha_scores=panel).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Parameter container + Adam state
# ---------------------------------------------------------------------------


class _Params:
    """LSTM + output-layer weight matrices, Xavier-initialized."""

    __slots__ = ("Wx", "Wh", "b", "Wy", "by")

    def __init__(self, input_size: int, hidden_size: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (input_size + hidden_size))

        # Combined gate matrices [i; f; g; o], each block (hidden_size, ?)
        self.Wx: np.ndarray = rng.normal(0.0, scale, (4 * hidden_size, input_size))
        self.Wh: np.ndarray = rng.normal(0.0, scale, (4 * hidden_size, hidden_size))
        self.b: np.ndarray = np.zeros(4 * hidden_size)
        # Initialize forget-gate bias to 1.0 to encourage long-range gradients
        self.b[hidden_size : 2 * hidden_size] = 1.0

        scale_out = np.sqrt(2.0 / hidden_size)
        self.Wy: np.ndarray = rng.normal(0.0, scale_out, (1, hidden_size))
        self.by: np.ndarray = np.zeros(1)


class _AdamState:
    """Adam optimizer first and second moment estimates."""

    def __init__(self, params: _Params) -> None:
        self.m: dict[str, np.ndarray] = {
            k: np.zeros_like(getattr(params, k))
            for k in ("Wx", "Wh", "b", "Wy", "by")
        }
        self.v: dict[str, np.ndarray] = {
            k: np.zeros_like(getattr(params, k))
            for k in ("Wx", "Wh", "b", "Wy", "by")
        }
        self.t: int = 0


def _adam_step(
    params: _Params,
    grads: dict[str, np.ndarray],
    adam: _AdamState,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> None:
    adam.t += 1
    t = adam.t
    bc1 = 1.0 - beta1**t
    bc2 = 1.0 - beta2**t
    for name in ("Wx", "Wh", "b", "Wy", "by"):
        g = grads[name]
        m, v = adam.m[name], adam.v[name]
        m[:] = beta1 * m + (1.0 - beta1) * g
        v[:] = beta2 * v + (1.0 - beta2) * g**2
        getattr(params, name)[:] -= lr * (m / bc1) / (np.sqrt(v / bc2) + eps)


# ---------------------------------------------------------------------------
# LSTM forward pass
# ---------------------------------------------------------------------------


def _lstm_forward(
    X: np.ndarray,  # (T, input_size)
    params: _Params,
) -> tuple[np.ndarray, list[dict]]:
    """Forward pass; returns (h_final, per-step caches) for BPTT."""
    T, _ = X.shape
    H = params.Wh.shape[1]  # hidden_size = cols of Wh

    h = np.zeros(H)
    c = np.zeros(H)
    caches: list[dict] = []

    for t in range(T):
        x = X[t]
        z = params.Wx @ x + params.Wh @ h + params.b  # (4H,)

        i_gate = _sigmoid(z[:H])
        f_gate = _sigmoid(z[H : 2 * H])
        g_gate = np.tanh(z[2 * H : 3 * H])
        o_gate = _sigmoid(z[3 * H :])

        c_new = f_gate * c + i_gate * g_gate
        h_new = o_gate * np.tanh(c_new)

        caches.append(
            {
                "x": x,
                "h_prev": h,
                "c_prev": c,
                "i": i_gate,
                "f": f_gate,
                "g": g_gate,
                "o": o_gate,
                "c": c_new,
                "h": h_new,
            }
        )
        h, c = h_new, c_new

    return h, caches


# ---------------------------------------------------------------------------
# BPTT backward pass
# ---------------------------------------------------------------------------


def _lstm_backward(
    dh_final: np.ndarray,  # (H,) gradient into last h from output layer
    caches: list[dict],
    params: _Params,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """BPTT; returns (dWx, dWh, db) — raw per-sample grads (no clipping here)."""
    H = dh_final.shape[0]
    dWx = np.zeros_like(params.Wx)
    dWh = np.zeros_like(params.Wh)
    db = np.zeros_like(params.b)

    dh = dh_final.copy()
    dc = np.zeros(H)  # gradient from c_{t+1} (none at t=T)

    for cache in reversed(caches):
        i, f, g, o = cache["i"], cache["f"], cache["g"], cache["o"]
        c_prev, c = cache["c_prev"], cache["c"]
        h_prev, x = cache["h_prev"], cache["x"]

        tanh_c = np.tanh(c)

        # Total gradient w.r.t. c_t from both h_t and c_{t+1}
        dc_total = dh * o * (1.0 - tanh_c**2) + dc

        # Gate gradients
        do = dh * tanh_c
        di = dc_total * g
        df = dc_total * c_prev
        dg = dc_total * i

        # Gradient to previous cell state
        dc = dc_total * f

        # Through activation functions
        dz = np.concatenate(
            [
                di * i * (1.0 - i),       # input gate
                df * f * (1.0 - f),       # forget gate
                dg * (1.0 - g**2),        # cell gate
                do * o * (1.0 - o),       # output gate
            ]
        )  # (4H,)

        dWx += np.outer(dz, x)
        dWh += np.outer(dz, h_prev)
        db += dz

        dh = params.Wh.T @ dz  # gradient to h_{t-1}

    return dWx, dWh, db


# ---------------------------------------------------------------------------
# Fit result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForecasterFitResult:
    """Training diagnostics from a single LSTMReturnForecaster.fit() call."""

    n_samples: int
    n_features: int
    n_epochs_run: int
    train_loss_history: list[float]

    @property
    def final_train_loss(self) -> float:
        return self.train_loss_history[-1]

    @property
    def initial_train_loss(self) -> float:
        return self.train_loss_history[0]


# ---------------------------------------------------------------------------
# Public forecaster
# ---------------------------------------------------------------------------


class LSTMReturnForecaster:
    """Cross-sectional return forecaster using a shared-weight LSTM.

    Training: for each (asset, date) pair, a lookback-length sequence of
    factor scores is used to predict the next-period return. The same LSTM
    weights are shared across all assets (treats assets as independent
    samples), giving more training data and implicit cross-asset transfer.

    Alpha generation: at rebalance date t, for each asset i run the LSTM
    on the last `lookback` periods of factor scores → scalar prediction.
    Cross-sectionally z-score predictions across assets → alpha signal for
    ResearchPipeline.

    All computation is pure numpy (no PyTorch / TensorFlow dependency).
    """

    def __init__(
        self,
        hidden_size: int = 32,
        lookback: int = 20,
        learning_rate: float = 5e-4,
        n_epochs: int = 50,
        batch_size: int = 64,
        grad_clip: float = 1.0,
        seed: int = 42,
        min_train_samples: int = 30,
    ) -> None:
        """
        Args:
            hidden_size: LSTM hidden state dimensionality.
            lookback: Sequence length (past periods) fed to the LSTM.
            learning_rate: Adam learning rate.
            n_epochs: Training epochs per fit call.
            batch_size: Mini-batch size.
            grad_clip: Global gradient L2-norm clip threshold (applied before
                the Adam update, averaged over the batch).
            seed: RNG seed for weight initialization and mini-batch shuffling.
            min_train_samples: Minimum sample count before fit() is allowed.
        """
        if hidden_size < 1:
            raise ValueError(f"hidden_size must be >= 1, got {hidden_size}.")
        if lookback < 1:
            raise ValueError(f"lookback must be >= 1, got {lookback}.")
        if learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}.")
        if n_epochs < 1:
            raise ValueError(f"n_epochs must be >= 1, got {n_epochs}.")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
        if min_train_samples < 1:
            raise ValueError(f"min_train_samples must be >= 1, got {min_train_samples}.")

        self.hidden_size = hidden_size
        self.lookback = lookback
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.seed = seed
        self.min_train_samples = min_train_samples

        self._params: _Params | None = None
        self._n_features: int | None = None
        self._x_mean: np.ndarray | None = None  # (1, 1, F)
        self._x_std: np.ndarray | None = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self._fitted: bool = False
        self.fit_result: ForecasterFitResult | None = None

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,  # (N, lookback, n_features)
        y: np.ndarray,  # (N,)
    ) -> LSTMReturnForecaster:
        """Fit on pre-windowed sequences.

        Args:
            X: Shape (N, T, F) where N = samples, T = lookback, F = features.
            y: Target next-period returns, shape (N,).
        """
        if X.ndim != 3:
            raise ValueError(f"X must be 3-D (N, T, F), got shape {X.shape}.")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be 1-D with len(y) == len(X).")

        n_samples, T, n_features = X.shape
        if T != self.lookback:
            raise ValueError(
                f"X.shape[1]={T} must match lookback={self.lookback}."
            )
        if n_samples < self.min_train_samples:
            raise ValueError(
                f"Need at least {self.min_train_samples} samples, got {n_samples}."
            )

        # Feature normalization (stored for inference)
        self._x_mean = X.mean(axis=(0, 1), keepdims=True)   # (1, 1, F)
        x_std = X.std(axis=(0, 1), keepdims=True)
        self._x_std = np.where(x_std < 1e-8, 1.0, x_std)
        X_norm = (X - self._x_mean) / self._x_std

        # Target normalization
        self._y_mean = float(y.mean())
        y_std = float(y.std())
        self._y_std = max(y_std, 1e-8)
        y_norm = (y - self._y_mean) / self._y_std

        # Re-initialize params when feature dimension changes
        if self._params is None or self._n_features != n_features:
            self._params = _Params(n_features, self.hidden_size, seed=self.seed)
            self._n_features = n_features

        adam = _AdamState(self._params)
        rng = np.random.default_rng(self.seed)
        loss_history: list[float] = []

        for _ in range(self.n_epochs):
            perm = rng.permutation(n_samples)
            epoch_losses: list[float] = []

            for start in range(0, n_samples, self.batch_size):
                batch = perm[start : start + self.batch_size]
                Xb = X_norm[batch]
                yb = y_norm[batch]
                B = len(batch)

                dWx_acc = np.zeros_like(self._params.Wx)
                dWh_acc = np.zeros_like(self._params.Wh)
                db_acc = np.zeros_like(self._params.b)
                dWy_acc = np.zeros_like(self._params.Wy)
                dby_acc = np.zeros_like(self._params.by)
                batch_loss = 0.0

                for k in range(B):
                    h_final, caches = _lstm_forward(Xb[k], self._params)
                    y_hat = float(np.dot(self._params.Wy[0], h_final) + self._params.by[0])
                    err = y_hat - float(yb[k])
                    batch_loss += err**2

                    # Output layer gradients
                    dy_hat = 2.0 * err
                    dWy_k = dy_hat * h_final[None, :]   # (1, H)
                    dby_k = np.array([dy_hat])
                    # Squeeze: Wy is (1, H), so Wy.T is (H, 1), need (H,)
                    dh_final = (self._params.Wy.T * dy_hat).squeeze()  # (H,)

                    dWx_k, dWh_k, db_k = _lstm_backward(dh_final, caches, self._params)
                    dWx_acc += dWx_k
                    dWh_acc += dWh_k
                    db_acc += db_k
                    dWy_acc += dWy_k
                    dby_acc += dby_k

                epoch_losses.append(batch_loss / B)

                # Average gradients and apply global norm clipping
                grads = {
                    "Wx": dWx_acc / B,
                    "Wh": dWh_acc / B,
                    "b": db_acc / B,
                    "Wy": dWy_acc / B,
                    "by": dby_acc / B,
                }
                total_norm = float(
                    np.sqrt(sum(float(np.sum(g**2)) for g in grads.values()))
                )
                if total_norm > self.grad_clip:
                    scale = self.grad_clip / total_norm
                    grads = {k: v * scale for k, v in grads.items()}

                _adam_step(self._params, grads, adam, self.learning_rate)

            loss_history.append(float(np.mean(epoch_losses)))

        self._fitted = True
        self.fit_result = ForecasterFitResult(
            n_samples=n_samples,
            n_features=n_features,
            n_epochs_run=self.n_epochs,
            train_loss_history=loss_history,
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Raw LSTM scalar output for each sample (not z-scored).

        Args:
            X: Shape (N, T, F). T must equal self.lookback.
        Returns:
            Shape (N,) raw predictions (in target return space).
        """
        if not self._fitted or self._params is None:
            raise RuntimeError("Call fit() before predict().")
        if X.ndim != 3:
            raise ValueError(f"X must be 3-D (N, T, F), got shape {X.shape}.")
        if self._x_mean is None or self._x_std is None:
            raise RuntimeError("Normalization stats not set; call fit() first.")

        X_norm = (X - self._x_mean) / self._x_std
        preds = []
        for k in range(len(X_norm)):
            h, _ = _lstm_forward(X_norm[k], self._params)
            raw = float(np.dot(self._params.Wy[0], h) + self._params.by[0])
            preds.append(raw * self._y_std + self._y_mean)
        return np.array(preds)

    def predict_zscore(self, X: np.ndarray) -> np.ndarray:
        """Cross-sectional z-score of predictions (directly usable as alpha)."""
        raw = self.predict(X)
        std = float(raw.std())
        if std < 1e-10:
            return np.zeros_like(raw)
        return (raw - float(raw.mean())) / std

    # ------------------------------------------------------------------
    # Sequence construction
    # ------------------------------------------------------------------

    def build_sequences(
        self,
        factor_panel: pd.DataFrame,  # (T × n_features) for ONE asset
        returns: pd.Series,           # next-period returns, same index
    ) -> tuple[np.ndarray, np.ndarray]:
        """Slide a lookback window to produce training sequences.

        Returns:
            X: (N, lookback, n_features)
            y: (N,) next-period returns
        """
        common = factor_panel.index.intersection(returns.index)
        fp = factor_panel.reindex(common).to_numpy(dtype=float)
        tgts = returns.reindex(common).to_numpy(dtype=float)
        n = len(fp)

        seqs: list[np.ndarray] = []
        ys: list[float] = []

        for t in range(self.lookback, n):
            window = fp[t - self.lookback : t]  # (lookback, F)
            seqs.append(window)
            ys.append(float(tgts[t]))

        if not seqs:
            n_f = factor_panel.shape[1]
            return np.empty((0, self.lookback, n_f), dtype=float), np.empty(0)
        return np.stack(seqs), np.array(ys)

    # ------------------------------------------------------------------
    # Alpha panel generation (walk-forward)
    # ------------------------------------------------------------------

    def generate_alpha_panel(
        self,
        factor_panels: dict[str, pd.DataFrame],  # asset → (T × F)
        returns: pd.DataFrame,                    # T × assets
        retrain_every: int = 21,
        min_train_periods: int = 252,
    ) -> pd.DataFrame:
        """Walk-forward alpha panel generation.

        At each rebalance date (every retrain_every periods once we have
        min_train_periods of history):
        1. Collect training data across all assets up to (excluding) that date.
        2. Fit the LSTM on the pooled (N_assets × N_windows) dataset.
        3. Run inference for each asset → cross-sectional z-score → alpha row.

        Returns:
            DataFrame (dates × assets) of z-score alpha signals.
        """
        assets = [a for a in factor_panels if a in returns.columns]
        all_dates = sorted(returns.index)
        alpha_rows: list[dict] = []

        for date_i, date in enumerate(all_dates):
            if date_i < min_train_periods:
                continue
            if date_i % retrain_every != 0 and self._fitted:
                # Inference only
                pass
            else:
                # Collect joint training set from all assets up to this date
                all_X: list[np.ndarray] = []
                all_y: list[np.ndarray] = []
                for asset in assets:
                    fp = factor_panels[asset]
                    fp_hist = fp[fp.index < date]
                    ret_hist = returns[asset][returns[asset].index < date]
                    if len(fp_hist) < self.lookback + self.min_train_samples:
                        continue
                    X_a, y_a = self.build_sequences(fp_hist, ret_hist)
                    if len(X_a) > 0:
                        all_X.append(X_a)
                        all_y.append(y_a)

                if all_X:
                    X_train = np.concatenate(all_X, axis=0)
                    y_train = np.concatenate(all_y, axis=0)
                    if len(X_train) >= self.min_train_samples:
                        self.fit(X_train, y_train)

            if not self._fitted:
                continue

            # Inference: one sequence per asset
            pred_assets: list[str] = []
            pred_X: list[np.ndarray] = []
            for asset in assets:
                fp = factor_panels[asset]
                fp_to = fp[fp.index <= date]
                if len(fp_to) < self.lookback:
                    continue
                seq = fp_to.iloc[-self.lookback:].to_numpy(dtype=float)
                pred_assets.append(asset)
                pred_X.append(seq)

            if len(pred_assets) < 2:
                continue

            X_pred = np.stack(pred_X)                # (N_assets, lookback, F)
            z_scores = self.predict_zscore(X_pred)   # (N_assets,)
            row: dict = {"date": date}
            for asset, z in zip(pred_assets, z_scores):
                row[asset] = float(z)
            alpha_rows.append(row)

        if not alpha_rows:
            return pd.DataFrame(columns=assets)

        df = pd.DataFrame(alpha_rows).set_index("date")
        return df.reindex(columns=assets)
