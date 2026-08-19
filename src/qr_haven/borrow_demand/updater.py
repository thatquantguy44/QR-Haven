"""Online streaming updates for the borrow demand surface.

GPyTorch's SVGP stores a variational posterior q(u) over m inducing outputs.
True rank-1 variational updates require re-running ELBO optimisation, which is
too slow for intraday ticks.

This module takes a practical two-layer approach:

  Layer 1 — Observation buffer: new locate-request events are buffered in memory.
  Layer 2 — Posterior correction: at query time, the GP posterior is conditioned
             on the buffer via exact GP conditioning restricted to the m inducing
             outputs, giving a low-rank update that costs O(m²) once per query.

Full model retraining (offline) is recommended at least daily.  The updater is
designed to bridge the gap between retraining cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import torch
import gpytorch

from qr_haven.borrow_demand.model import BorrowDemandSurface
from qr_haven.borrow_demand.features import SurfaceFeatures, N_FEATURES


@dataclass
class LocateEvent:
    """A single intraday locate-request observation for streaming update."""

    event_timestamp: datetime
    cusip: str
    features: SurfaceFeatures       # normalised features at time of request
    demand_shares: float            # requested quantity (training signal)


class OnlineSurfaceUpdater:
    """Buffer intraday locate events; correct GP posterior at query time.

    Parameters
    ----------
    model : BorrowDemandSurface
        The fitted (eval-mode) SVGP model.
    likelihood : GaussianLikelihood
        The fitted likelihood with learned noise variance.
    max_buffer_size : int
        Maximum number of intraday events to retain.  Oldest events are
        dropped when the buffer is full (FIFO).
    """

    def __init__(
        self,
        model: BorrowDemandSurface,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        max_buffer_size: int = 1000,
    ) -> None:
        self.model = model
        self.likelihood = likelihood
        self.max_buffer_size = max_buffer_size
        self._buffer: list[LocateEvent] = []

    def ingest(self, event: LocateEvent) -> None:
        """Add a new locate-request event to the intraday buffer."""
        self._buffer.append(event)
        if len(self._buffer) > self.max_buffer_size:
            self._buffer.pop(0)

    def ingest_batch(self, events: list[LocateEvent]) -> None:
        for event in events:
            self.ingest(event)

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    def clear_buffer(self) -> None:
        """Clear all buffered events.  Call after a model retrain."""
        self._buffer.clear()

    @torch.no_grad()
    def predict(
        self,
        query_features: SurfaceFeatures,
    ) -> tuple[float, float]:
        """Posterior predictive mean and std at query_features.

        If the buffer is non-empty, corrects the base GP posterior using the
        buffered observations via exact conditioning (low-rank update).

        Returns
        -------
        mean : float
            Posterior mean demand estimate.
        std : float
            Posterior standard deviation.
        """
        use_time = getattr(self.model, "use_time_kernel", False)
        _to_t = lambda sf: sf.to_full_tensor() if use_time else sf.to_tensor()  # noqa: E731

        x_query = _to_t(query_features).unsqueeze(0)

        if len(self._buffer) == 0:
            mean_arr, std_arr = self.model.predict(x_query, self.likelihood)
            return float(mean_arr[0]), float(std_arr[0])

        # Build buffer arrays
        X_buf = torch.stack([_to_t(e.features) for e in self._buffer])
        y_buf = torch.tensor(
            [e.demand_shares for e in self._buffer], dtype=torch.float32
        )

        # Condition the fitted GP on buffer observations via exact GP conditioning
        # using the base model's prior + noise.
        self.model.eval()
        self.likelihood.eval()

        with gpytorch.settings.fast_pred_var():
            # Get the base model distribution at buffer points
            f_buf = self.model(X_buf)
            cov_buf = f_buf.covariance_matrix
            # GPyTorch ≥1.10 returns a LinearOperator; earlier returns a Tensor
            K_buf_buf = cov_buf.to_dense() if hasattr(cov_buf, "to_dense") else cov_buf
            sigma_n2 = self.likelihood.noise.item()
            # K_buf_buf + σ²I
            K_noise = K_buf_buf + sigma_n2 * torch.eye(len(self._buffer))

            # Cross-covariance: query vs buffer
            K_q_buf_lazy = self.model.covar_module(x_query, X_buf)
            K_q_buf = (
                K_q_buf_lazy.to_dense()
                if hasattr(K_q_buf_lazy, "to_dense")
                else K_q_buf_lazy.evaluate()
            )  # (1, m_buf)

            # GP residuals from base model mean
            mu_buf = self.model.mean_module(X_buf)  # (m_buf,)
            residuals = y_buf - mu_buf              # (m_buf,)

            # Posterior mean correction
            L = torch.linalg.cholesky(K_noise)
            alpha = torch.cholesky_solve(residuals.unsqueeze(1), L).squeeze(1)
            delta_mean = float((K_q_buf @ alpha).item())

            # Posterior variance correction
            v = torch.linalg.solve_triangular(L, K_q_buf.T, upper=False)  # (m_buf, 1)
            delta_var = float((v.T @ v).item())

            # Base posterior
            base_mean_arr, base_std_arr = self.model.predict(x_query, self.likelihood)
            base_mean = float(base_mean_arr[0])
            base_var = float(base_std_arr[0]) ** 2

        corrected_mean = base_mean + delta_mean
        corrected_var = max(base_var - delta_var, 1e-8)  # clamp to avoid negative
        return corrected_mean, float(np.sqrt(corrected_var))
