import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from models.kan_transport import MonotonicKANMap, ScaleModulatedKANMap
from models.root_finding import invert_triangular_map


class SMKANDensityEstimator(nn.Module):
    """
    Scale-Modulated Monotonic Kolmogorov-Arnold Network (SM-KAN) Autoregressive Density Estimator.

    Implements a lower-triangular Knothe-Rosenblatt rearrangement transport map S: R^d -> R^d:
        S(x) = [ S^1(x_1), S^2(x_1, x_2), ..., S^d(x_1, ..., x_d) ]^T

    Each coordinate map S^k: R^k -> R pushes forward the conditional measure pi(x_k | x_{<k})
    to standard normal reference N(0, 1) using ScaleModulatedKANMap:
        S^k(x_{<k}, x_k) = exp(s_max * tanh(sigma(x_{<k}) / s_max)) * S^k_core(x_{<k}, x_k) + alpha(x_{<k})

    Exact closed-form log-density via Knothe-Rosenblatt lower-triangular Jacobian determinant:
        log pi_S(x) = - (d / 2) * log(2 * pi) - 0.5 * ||S(x)||_2^2 + sum_{k=1}^d log(partial_k S^k(x_{<k}, x_k))

    Generative inversion S^{-1}(z) uses fast batched 1D Newton-Raphson with guaranteed convergence
    in 3-5 iterations (no numerical ODE integration or Picard iteration).
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 4,
        grid_size: int = 8,
        grid_range: Tuple[float, float] = (-4.0, 4.0),
        spline_order: int = 3,
        conditioning_scale: bool = True,
        scale_bound: float = 4.0,
        penalty_type: str = "functional",
        dev: str = "cpu",
    ):
        """
        Args:
            num_features: Dimensionality d of the data distribution.
            hidden_dim: Hidden dimension Q in the KAN layers.
            grid_size: Number of B-spline grid intervals.
            grid_range: (grid_min, grid_max) domain for B-spline knots.
            spline_order: Degree p of B-spline basis (3 for cubic splines).
            conditioning_scale: Whether to apply smooth scale modulation exp(sigma) * S_core + alpha.
            scale_bound: Maximum saturation bound s_max for scale modulation.
            penalty_type: Regularization penalty ('functional', 'coeff_group', or 'target_jacobian').
            dev: Device string ('cpu' or 'cuda').
        """
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.grid_range = grid_range
        self.spline_order = spline_order
        self.conditioning_scale = conditioning_scale
        self.scale_bound = float(scale_bound)
        self.penalty_type = penalty_type
        self.device = torch.device(dev)

        # Standardizing shift & scale buffers (optional, identity by default)
        self.register_buffer("data_mean", torch.zeros(num_features, device=self.device))
        self.register_buffer("data_std", torch.ones(num_features, device=self.device))
        self.use_standardization = False

        maps = []
        for k in range(num_features):
            map_cls = ScaleModulatedKANMap if conditioning_scale else MonotonicKANMap
            m = map_cls(
                num_features=k + 1,
                target_index=k,
                hidden_dim=hidden_dim,
                grid_size=grid_size,
                grid_range=grid_range,
                spline_order=spline_order,
                penalty_type=penalty_type,
                scale_bound=scale_bound,
                dev=dev,
            )
            maps.append(m)
        self.maps = nn.ModuleList(maps)

    def set_standardization(self, mean: torch.Tensor, std: torch.Tensor):
        """
        Sets data standardization parameters: u = (x - mean) / std.
        Accounts for log-determinant correction -sum log(std).
        """
        self.data_mean.copy_(mean.to(self.device))
        self.data_std.copy_(std.to(self.device).clamp(min=1e-6))
        self.use_standardization = True

    def _normalize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.use_standardization:
            return x, torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        u = (x - self.data_mean) / self.data_std
        log_det_norm = -torch.sum(torch.log(self.data_std)).expand(x.shape[0])
        return u, log_det_norm

    def _unnormalize(self, u: torch.Tensor) -> torch.Tensor:
        if not self.use_standardization:
            return u
        return u * self.data_std + self.data_mean

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pushforward mapping x -> z = S(x) and exact log-determinant of Jacobian.

        Args:
            x: Input tensor of shape (B, d).

        Returns:
            z: Latent standard normal vectors of shape (B, d).
            log_det_j: Total log-determinant log |det nabla S(x)| of shape (B,).
        """
        u, log_det_norm = self._normalize(x)
        B, d = u.shape

        z = torch.empty((B, d), device=u.device, dtype=u.dtype)
        log_det_j = log_det_norm.clone()

        for k in range(d):
            u_k_all = u[:, : k + 1]
            derivs = self.maps[k].compute_derivatives(u_k_all)
            z[:, k] = derivs["out"].squeeze(-1)
            target_deriv = derivs["S_k"].squeeze(-1)
            log_det_j = log_det_j + torch.log(target_deriv)

        return z, log_det_j

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the exact log-likelihood log pi_S(x) under the change of variables formula:
            log pi_S(x) = log p_Z(S(x)) + log |det nabla S(x)|

        Args:
            x: Input tensor of shape (B, d).

        Returns:
            log_density: Tensor of shape (B,) containing exact point log-densities in nats.
        """
        B, d = x.shape
        z, log_det_j = self.forward(x)
        log_pz = -0.5 * (d * math.log(2.0 * math.pi) + torch.sum(z**2, dim=-1))
        return log_pz + log_det_j

    def compute_loss(self, x: torch.Tensor, reg_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the regularized negative log-likelihood (NLL) loss over batch x:
            Loss = - (1/B) sum_{i=1}^B log pi_S(x^{(i)}) + reg_value * sum_{k=1}^d Penalty(S^k)

        Args:
            x: Training batch of shape (B, d).
            reg_value: Regularization hyperparameter coefficient.

        Returns:
            total_loss: Total scalar loss for backpropagation.
            nll_loss: Unregularized average negative log-likelihood (nats).
            reg_penalty: Summed regularization penalty.
        """
        u, log_det_norm = self._normalize(x)
        B, d = u.shape

        total_nll = torch.tensor(0.5 * d * math.log(2.0 * math.pi), device=x.device, dtype=x.dtype)
        total_nll = total_nll - log_det_norm.mean()
        total_reg = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        for k in range(d):
            u_k_all = u[:, : k + 1]
            derivs = self.maps[k].compute_derivatives(u_k_all)
            out = derivs["out"]
            S_k = derivs["S_k"]
            S_all = derivs["S_all"]

            coord_nll = (0.5 * (out**2) - torch.log(S_k)).mean()
            total_nll = total_nll + coord_nll

            if reg_value > 0.0:
                coord_reg = self.maps[k].compute_regularization(S_all)
                total_reg = total_reg + coord_reg

        total_loss = total_nll + reg_value * total_reg
        return total_loss, total_nll, total_reg

    def invert(
        self,
        z: torch.Tensor,
        max_iter: int = 30,
        tol: float = 1e-6,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Exact generative inversion: solves S(u) = z for u coordinate-by-coordinate,
        then transforms back to original space x = u * std + mean.

        Args:
            z: Latent samples of shape (B, d).
            max_iter: Maximum Newton iterations per coordinate.
            tol: Inversion tolerance ||S^k(u) - z_k|| < tol.
            device: Target device.

        Returns:
            x: Inverted samples in data space of shape (B, d).
            diagnostics: List of per-coordinate root-finding diagnostic dictionaries.
        """
        dev = self.device if device is None else torch.device(device)
        u, diagnostics = invert_triangular_map(
            maps=self.maps,
            z=z,
            max_iter=max_iter,
            tol=tol,
            device=dev,
        )
        x = self._unnormalize(u)
        return x, diagnostics

    def sample(
        self,
        n_samples: int,
        max_iter: int = 30,
        tol: float = 1e-6,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Generates new samples from the learned distribution:
            1. Sample z ~ N(0, I_d)
            2. Solve S(u) = z via Newton-Raphson
            3. Return x = unnormalize(u)

        Args:
            n_samples: Number of samples to generate.
            max_iter: Maximum Newton iterations per coordinate.
            tol: Inversion convergence tolerance.
            device: Target device.

        Returns:
            samples: Generated data samples of shape (n_samples, d).
        """
        dev = self.device if device is None else torch.device(device)
        z = torch.randn((n_samples, self.num_features), device=dev)
        x, _ = self.invert(z, max_iter=max_iter, tol=tol, device=dev)
        return x
