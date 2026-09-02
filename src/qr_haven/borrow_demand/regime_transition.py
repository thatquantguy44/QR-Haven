"""Borrow Regime Transition Model.

Predicts P(regime(t+k) = r' | regime(t) = r, X(t)) for k in {1, 5, 10, 20}.

Two architectures:
  A — EmpiricalTransitionMatrix  : lookup table stratified by earnings proximity / sector
  B — LogisticTransitionForecaster: per-(from_r, to_r, k) logistic regression with borrow features
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from qr_haven.borrow_demand.earnings_forecast import EarningsRecord

__all__ = [
    "BorrowRegime",
    "RegimeTransitionResult",
    "EmpiricalTransitionMatrix",
    "LogisticTransitionForecaster",
    "BorrowRegimeClassifier",
]

# ---------------------------------------------------------------------------
# Regime enum + thresholds (bps)
# ---------------------------------------------------------------------------

GC_MAX_BPS: float = 25.0
WARM_MAX_BPS: float = 150.0

_HORIZONS: Tuple[int, ...] = (1, 5, 10, 20)
_N_REGIMES: int = 3


class BorrowRegime(enum.IntEnum):
    GC = 0    # < 25 bps
    WARM = 1  # 25 – 150 bps
    HTB = 2   # > 150 bps


def _fee_to_regime(fee_bps: float) -> BorrowRegime:
    if fee_bps < GC_MAX_BPS:
        return BorrowRegime.GC
    if fee_bps < WARM_MAX_BPS:
        return BorrowRegime.WARM
    return BorrowRegime.HTB


def label_series(fee_series: np.ndarray) -> np.ndarray:
    """Vectorised label assignment. Returns int array with BorrowRegime values."""
    out = np.full(len(fee_series), BorrowRegime.GC, dtype=np.int8)
    out[fee_series >= GC_MAX_BPS] = BorrowRegime.WARM
    out[fee_series >= WARM_MAX_BPS] = BorrowRegime.HTB
    return out


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RegimeTransitionResult:
    """Predicted transition probabilities.

    probs[k_idx][to_regime] = P(regime(t+k) = to_regime | regime(t), X(t))
    where k_idx corresponds to _HORIZONS = (1, 5, 10, 20).
    """

    from_regime: BorrowRegime
    horizons: Tuple[int, ...] = _HORIZONS
    # shape: (n_horizons, n_regimes)
    probs: np.ndarray = field(default_factory=lambda: np.zeros((4, 3)))

    def prob_at(self, horizon: int, to_regime: BorrowRegime) -> float:
        k_idx = list(self.horizons).index(horizon)
        return float(self.probs[k_idx, int(to_regime)])

    def most_likely_at(self, horizon: int) -> BorrowRegime:
        k_idx = list(self.horizons).index(horizon)
        return BorrowRegime(int(np.argmax(self.probs[k_idx])))


# ---------------------------------------------------------------------------
# Architecture A — EmpiricalTransitionMatrix
# ---------------------------------------------------------------------------

class EmpiricalTransitionMatrix:
    """Empirical transition matrix T[from_r, to_r, k_idx].

    Optionally stratified by:
      - earnings proximity bucket: 'near' (|days_to_earnings| <= 5) or 'far'
      - sector code (string)
    """

    def __init__(self, pseudocount: float = 0.5) -> None:
        self._pseudocount = pseudocount
        # raw counts: dict[(bucket, sector)] -> ndarray(n_r, n_r, n_k)
        self._counts: Dict[Tuple[str, str], np.ndarray] = {}
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(
        self,
        fee_series: np.ndarray,
        days_to_earnings: Optional[np.ndarray] = None,
        sectors: Optional[Sequence[str]] = None,
    ) -> "EmpiricalTransitionMatrix":
        """Count transitions across all horizons.

        Parameters
        ----------
        fee_series      : 1-D float array of daily borrow fees (bps), chronological
        days_to_earnings: integer array of same length (0 = earnings day, negative = before)
        sectors         : string array of sector labels (same length), or None
        """
        n = len(fee_series)
        regimes = label_series(fee_series)

        if days_to_earnings is None:
            days_to_earnings = np.full(n, 999, dtype=int)
        if sectors is None:
            sectors = ["_all"] * n

        for t in range(n):
            bucket = "near" if abs(days_to_earnings[t]) <= 5 else "far"
            sector = sectors[t]
            key = (bucket, sector)
            if key not in self._counts:
                self._counts[key] = np.zeros((_N_REGIMES, _N_REGIMES, len(_HORIZONS)))

            r_from = regimes[t]
            for ki, k in enumerate(_HORIZONS):
                if t + k < n:
                    r_to = regimes[t + k]
                    self._counts[key][r_from, r_to, ki] += 1.0

        self._fitted = True
        return self

    def _get_matrix(self, bucket: str, sector: str) -> np.ndarray:
        """Return (n_r, n_r, n_k) probability matrix with pseudocounts."""
        # Collect matching strata
        candidates = []
        for (b, s), cnt in self._counts.items():
            if b == bucket and s == sector:
                candidates.append(cnt)
        if not candidates:
            # fall back to same bucket any sector
            for (b, s), cnt in self._counts.items():
                if b == bucket:
                    candidates.append(cnt)
        if not candidates:
            # full fallback — all data
            candidates = list(self._counts.values())

        total = np.zeros((_N_REGIMES, _N_REGIMES, len(_HORIZONS)))
        for c in candidates:
            total += c

        # Add pseudocounts and normalise
        total += self._pseudocount
        row_sums = total.sum(axis=1, keepdims=True)  # (n_r, 1, n_k)
        return total / np.maximum(row_sums, 1e-12)

    def predict(
        self,
        from_regime: BorrowRegime,
        days_to_earnings: int = 999,
        sector: str = "_all",
    ) -> np.ndarray:
        """Return (n_horizons, n_regimes) probability matrix."""
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict()")
        bucket = "near" if abs(days_to_earnings) <= 5 else "far"
        mat = self._get_matrix(bucket, sector)  # (n_r, n_r, n_k)
        # probs[k_idx, to_r] = mat[from_r, to_r, k_idx]
        return mat[int(from_regime), :, :].T  # (n_k, n_r)


# ---------------------------------------------------------------------------
# Architecture B — LogisticTransitionForecaster
# ---------------------------------------------------------------------------

_FEATURE_NAMES = (
    "delta_util_3d",
    "delta_si_proxy",
    "days_to_earnings",
    "delta_etf_pct",
    "fee_vol_5d",
    "sector_borrow_index",
)
_N_FEATURES = len(_FEATURE_NAMES)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class _LogisticOVR:
    """One-vs-rest logistic regression via gradient descent (numpy only)."""

    def __init__(self, n_classes: int, lr: float = 0.01, n_epochs: int = 200) -> None:
        self._n_classes = n_classes
        self._lr = lr
        self._n_epochs = n_epochs
        self._W: Optional[np.ndarray] = None  # (n_classes, n_features+1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_LogisticOVR":
        n, d = X.shape
        Xb = np.column_stack([X, np.ones(n)])  # bias column
        self._W = np.zeros((self._n_classes, d + 1))
        for _ in range(self._n_epochs):
            logits = Xb @ self._W.T           # (n, n_classes)
            probs = _sigmoid(logits)
            for c in range(self._n_classes):
                yc = (y == c).astype(float)
                grad = Xb.T @ (probs[:, c] - yc) / n
                self._W[c] -= self._lr * grad
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (n_samples, n_classes) probability matrix."""
        if self._W is None:
            raise RuntimeError("Not fitted")
        n = X.shape[0]
        Xb = np.column_stack([X, np.ones(n)])
        logits = Xb @ self._W.T
        p = _sigmoid(logits)
        row_sums = p.sum(axis=1, keepdims=True)
        return p / np.maximum(row_sums, 1e-12)


class LogisticTransitionForecaster:
    """Per-(from_regime, horizon) one-vs-rest classifier."""

    def __init__(self, lr: float = 0.05, n_epochs: int = 300) -> None:
        self._lr = lr
        self._n_epochs = n_epochs
        # models[(from_r, ki)] -> _LogisticOVR
        self._models: Dict[Tuple[int, int], _LogisticOVR] = {}
        self._feat_mean: Optional[np.ndarray] = None
        self._feat_std: Optional[np.ndarray] = None
        self._fitted = False

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self._feat_mean) / np.maximum(self._feat_std, 1e-8)

    def fit(
        self,
        fee_series: np.ndarray,
        features: np.ndarray,
    ) -> "LogisticTransitionForecaster":
        """Train classifiers.

        Parameters
        ----------
        fee_series : 1-D daily borrow fee (bps), length T
        features   : (T, n_features) array in order of _FEATURE_NAMES
        """
        n = len(fee_series)
        regimes = label_series(fee_series)

        self._feat_mean = features.mean(axis=0)
        self._feat_std = features.std(axis=0)
        Xs = self._standardize(features)

        for ki, k in enumerate(_HORIZONS):
            valid = n - k
            if valid < 2:
                continue
            Xv = Xs[:valid]
            y_from = regimes[:valid]
            y_to = regimes[k : n]

            for r in range(_N_REGIMES):
                mask = y_from == r
                if mask.sum() < 2:
                    continue
                clf = _LogisticOVR(_N_REGIMES, lr=self._lr, n_epochs=self._n_epochs)
                clf.fit(Xv[mask], y_to[mask])
                self._models[(r, ki)] = clf

        self._fitted = True
        return self

    def predict(
        self,
        from_regime: BorrowRegime,
        feature_vec: np.ndarray,
    ) -> np.ndarray:
        """Return (n_horizons, n_regimes) probability array."""
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict()")
        if self._feat_mean is None:
            raise RuntimeError("Scaler not initialised")
        x = self._standardize(feature_vec.reshape(1, -1))
        probs = np.full((len(_HORIZONS), _N_REGIMES), 1.0 / _N_REGIMES)
        for ki in range(len(_HORIZONS)):
            key = (int(from_regime), ki)
            if key in self._models:
                probs[ki] = self._models[key].predict_proba(x)[0]
        return probs


# ---------------------------------------------------------------------------
# Top-level: BorrowRegimeClassifier
# ---------------------------------------------------------------------------

@dataclass
class _FitState:
    empirical: EmpiricalTransitionMatrix
    logistic: LogisticTransitionForecaster
    ensemble_weight: float  # weight on logistic vs empirical (0 = pure empirical)


class BorrowRegimeClassifier:
    """Orchestrates both architectures.

    Usage
    -----
    clf = BorrowRegimeClassifier()
    clf.fit(fee_df, feature_df)
    result = clf.predict_transition(current_regime=BorrowRegime.HTB, feature_vec=x)
    """

    def __init__(
        self,
        ensemble_weight: float = 0.4,
        pseudocount: float = 0.5,
        lr: float = 0.05,
        n_epochs: int = 300,
    ) -> None:
        self._ensemble_weight = ensemble_weight
        self._pseudocount = pseudocount
        self._lr = lr
        self._n_epochs = n_epochs
        self._state: Optional[_FitState] = None

    @staticmethod
    def label_regime(fee_bps: float) -> BorrowRegime:
        return _fee_to_regime(fee_bps)

    @staticmethod
    def label_series(fee_series: np.ndarray) -> np.ndarray:
        return label_series(fee_series)

    def fit(
        self,
        fee_series: np.ndarray,
        features: np.ndarray,
        days_to_earnings: Optional[np.ndarray] = None,
        sectors: Optional[Sequence[str]] = None,
    ) -> "BorrowRegimeClassifier":
        """Fit both architectures.

        Parameters
        ----------
        fee_series       : 1-D daily borrow fees (bps), chronological
        features         : (T, n_features) array aligned to fee_series
        days_to_earnings : integer array (0 = event day); None → treat all as 'far'
        sectors          : string sector labels; None → treat all as '_all'
        """
        emp = EmpiricalTransitionMatrix(pseudocount=self._pseudocount)
        emp.fit(fee_series, days_to_earnings=days_to_earnings, sectors=sectors)

        log = LogisticTransitionForecaster(lr=self._lr, n_epochs=self._n_epochs)
        log.fit(fee_series, features)

        self._state = _FitState(
            empirical=emp,
            logistic=log,
            ensemble_weight=self._ensemble_weight,
        )
        return self

    def predict_transition(
        self,
        current_regime: BorrowRegime,
        feature_vec: np.ndarray,
        days_to_earnings: int = 999,
        sector: str = "_all",
    ) -> RegimeTransitionResult:
        """Return blended transition probabilities.

        Returns
        -------
        RegimeTransitionResult with probs shape (n_horizons, n_regimes).
        """
        if self._state is None:
            raise RuntimeError("Must call fit() before predict_transition()")

        p_emp = self._state.empirical.predict(
            current_regime,
            days_to_earnings=days_to_earnings,
            sector=sector,
        )
        p_log = self._state.logistic.predict(current_regime, feature_vec)

        w = self._state.ensemble_weight
        blended = (1.0 - w) * p_emp + w * p_log
        # Renormalise rows to sum to 1
        row_sums = blended.sum(axis=1, keepdims=True)
        blended = blended / np.maximum(row_sums, 1e-12)

        return RegimeTransitionResult(
            from_regime=current_regime,
            horizons=_HORIZONS,
            probs=blended,
        )
