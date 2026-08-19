"""Sparse Variational Gaussian Process surface for borrow demand.

Architecture:
  - ApproximateGP with CholeskyVariationalDistribution over m inducing points
  - ARD Matérn-5/2 kernel over [σ, ADV, SI, ETF, RQ] ∈ [0,1]^5
  - Linear mean function (warm-started from OLS)
  - GaussianLikelihood for observation noise
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

from qr_haven.borrow_demand.features import N_FEATURES


@dataclass
class BorrowDemandConfig:
    """Hyperparameters and training settings for BorrowDemandSurface."""

    n_inducing: int = 500        # number of inducing points (m ≪ n)
    matern_nu: float = 2.5       # Matérn smoothness; 2.5 = twice-differentiable
    lr: float = 0.01             # Adam learning rate for ELBO optimisation
    n_epochs: int = 100          # outer training epochs
    batch_size: int = 256        # mini-batch size for stochastic ELBO
    noise_init: float = 1.0      # initial observation noise variance
    learn_inducing: bool = True  # whether inducing locations are also optimised

    def __post_init__(self) -> None:
        if self.n_inducing < 10:
            raise ValueError("n_inducing must be ≥ 10")
        if self.matern_nu not in (0.5, 1.5, 2.5):
            raise ValueError("matern_nu must be 0.5, 1.5, or 2.5")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.n_epochs < 1:
            raise ValueError("n_epochs must be ≥ 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be ≥ 1")


class BorrowDemandSurface(ApproximateGP):
    """SVGP borrow-demand surface model.

    Input:  x ∈ [0,1]^5  (SurfaceFeatures percentile ranks)
    Output: D(x) ~ GP(μ(x), k(x,x'))

    Kernel:  k = ScaleKernel(Matérn-5/2, ARD)
    Mean:    μ = LinearMean  (affine in x; captures monotone trends)

    The ScaleKernel wraps the Matérn to learn an output scale σ_f separately
    from the ARD lengthscales ℓ_j, which gate feature relevance.
    """

    def __init__(self, inducing_points: torch.Tensor) -> None:
        if inducing_points.dim() != 2 or inducing_points.size(1) != N_FEATURES:
            raise ValueError(
                f"inducing_points must be (m, {N_FEATURES}); "
                f"got {tuple(inducing_points.shape)}"
            )
        m = inducing_points.size(0)
        variational_distribution = CholeskyVariationalDistribution(m)
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.LinearMean(input_size=N_FEATURES)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=N_FEATURES)
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) of the posterior predictive distribution.

        mean and std are numpy arrays matching x.shape[0].
        """
        self.eval()
        likelihood.eval()
        with gpytorch.settings.fast_pred_var():
            pred = likelihood(self(x))
        mean = pred.mean.cpu().numpy()
        std = pred.variance.sqrt().cpu().numpy()
        return mean, std

    @staticmethod
    def init_inducing_kmeans(
        X: np.ndarray,
        n_inducing: int,
        seed: int = 0,
    ) -> torch.Tensor:
        """Initialise inducing points via k-means++ on the training feature matrix.

        X: (n, N_FEATURES) float array of normalised features.
        Returns a (min(n_inducing, n), N_FEATURES) float32 tensor.
        """
        from sklearn.cluster import MiniBatchKMeans

        m = min(n_inducing, len(X))
        kmeans = MiniBatchKMeans(n_clusters=m, random_state=seed, n_init=3)
        kmeans.fit(X)
        centers = kmeans.cluster_centers_.astype(np.float32)
        return torch.tensor(centers)
