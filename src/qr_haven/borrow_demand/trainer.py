"""Offline ELBO training loop for BorrowDemandSurface.

Trains the SVGP by maximising the Evidence Lower BOund (ELBO) with
mini-batch stochastic gradient descent.  Returns a TrainingResult with
per-epoch losses and fitted model state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import gpytorch
from torch.utils.data import DataLoader, TensorDataset
from gpytorch.mlls import VariationalELBO

from qr_haven.borrow_demand.model import BorrowDemandConfig, BorrowDemandSurface


@dataclass
class TrainingResult:
    """Summary returned by train_surface."""

    epoch_losses: list[float]     # mean ELBO loss per epoch (negative → lower is better)
    final_loss: float             # last epoch loss
    n_epochs_run: int
    converged: bool               # True if loss improvement < tol over last 5 epochs
    noise_variance: float         # fitted observation noise σ²_n
    outputscale: float            # fitted kernel output scale σ²_f


def train_surface(
    model: BorrowDemandSurface,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    X_train: np.ndarray | torch.Tensor,
    y_train: np.ndarray | torch.Tensor,
    config: BorrowDemandConfig | None = None,
    tol: float = 1e-4,
    verbose: bool = False,
) -> TrainingResult:
    """Train the SVGP surface via ELBO maximisation.

    Parameters
    ----------
    model : BorrowDemandSurface
        Uninitialised or previously trained SVGP model.
    likelihood : GaussianLikelihood
        Observation noise model; jointly optimised with the GP.
    X_train : array (n, 5)
        Normalised feature matrix (SurfaceFeatures.to_array() stacked).
    y_train : array (n,)
        Demand target values (demand_shares, raw or log-transformed).
    config : BorrowDemandConfig, optional
        Training hyperparameters.  Defaults to BorrowDemandConfig().
    tol : float
        Convergence tolerance on mean epoch loss improvement.
    verbose : bool
        Print loss every 10 epochs.
    """
    cfg = config or BorrowDemandConfig()

    if isinstance(X_train, np.ndarray):
        X_train = torch.tensor(X_train, dtype=torch.float32)
    if isinstance(y_train, np.ndarray):
        y_train = torch.tensor(y_train, dtype=torch.float32)

    from qr_haven.borrow_demand.features import N_FEATURES, N_FULL_INPUT_DIMS
    expected_cols = N_FULL_INPUT_DIMS if cfg.use_time_kernel else N_FEATURES
    if X_train.dim() != 2 or X_train.size(1) != expected_cols:
        raise ValueError(
            f"X_train must be (n, {expected_cols}); got {tuple(X_train.shape)}"
        )
    if y_train.dim() != 1 or y_train.size(0) != X_train.size(0):
        raise ValueError("y_train must be 1-D with length matching X_train rows")

    n = X_train.size(0)
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": likelihood.parameters()}],
        lr=cfg.lr,
    )
    mll = VariationalELBO(likelihood, model, num_data=n)

    epoch_losses: list[float] = []
    for epoch in range(cfg.n_epochs):
        batch_losses: list[float] = []
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        mean_loss = float(np.mean(batch_losses))
        epoch_losses.append(mean_loss)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch + 1}/{cfg.n_epochs}  loss={mean_loss:.4f}")

    model.eval()
    likelihood.eval()

    # Convergence: improvement in last 5 epochs < tol
    converged = False
    if len(epoch_losses) >= 5:
        recent_improvement = abs(epoch_losses[-5] - epoch_losses[-1])
        converged = recent_improvement < tol

    noise_var = float(likelihood.noise.item())
    outputscale = float(model.covar_module.outputscale.item())

    return TrainingResult(
        epoch_losses=epoch_losses,
        final_loss=epoch_losses[-1],
        n_epochs_run=cfg.n_epochs,
        converged=converged,
        noise_variance=noise_var,
        outputscale=outputscale,
    )


def make_model_and_likelihood(
    X_train: np.ndarray,
    config: BorrowDemandConfig | None = None,
    seed: int = 0,
) -> tuple[BorrowDemandSurface, gpytorch.likelihoods.GaussianLikelihood]:
    """Convenience factory: initialise model + likelihood ready for training.

    Inducing points are placed by k-means++ on X_train.
    """
    cfg = config or BorrowDemandConfig()
    inducing = BorrowDemandSurface.init_inducing_kmeans(X_train, cfg.n_inducing, seed)
    model = BorrowDemandSurface(
        inducing,
        learn_inducing=cfg.learn_inducing,
        use_time_kernel=cfg.use_time_kernel,
    )
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = torch.tensor(cfg.noise_init)
    return model, likelihood
