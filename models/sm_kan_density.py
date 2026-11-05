import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from models.kan_transport import MonotonicKANMap, ScaleModulatedKANMap
from models.root_finding import invert_triangular_map


class SMKANDensityBlock(nn.Module):
    """
    Single Knothe-Rosenblatt triangular block S: R^d -> R^d consisting of d
    ScaleModulatedKANMap component maps S^k(x_{<k}, x_k).
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

    def forward(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, d = u.shape
        z = torch.empty((B, d), device=u.device, dtype=u.dtype)
        log_det_j = torch.zeros(B, device=u.device, dtype=u.dtype)

        for k in range(d):
            u_k_all = u[:, : k + 1]
            derivs = self.maps[k].compute_derivatives(u_k_all)
            z[:, k] = derivs["out"].squeeze(-1)
            target_deriv = derivs["S_k"].squeeze(-1)
            log_det_j = log_det_j + torch.log(target_deriv)

        return z, log_det_j

    def invert(self, z: torch.Tensor, max_iter: int = 30, tol: float = 1e-6) -> Tuple[torch.Tensor, List[Dict]]:
        return invert_triangular_map(
            maps=self.maps,
            z=z,
            max_iter=max_iter,
            tol=tol,
            device=self.device,
        )


class SMKANDensityEstimator(nn.Module):
    """
    Scale-Modulated Monotonic Kolmogorov-Arnold Network (SM-KAN) Autoregressive Density Estimator
    and Deep Composite Normalizing Flow.

    Architecture Options:
        1. Single Triangular Map (num_steps=1):
           A lower-triangular Knothe-Rosenblatt rearrangement transport map S: R^d -> R^d:
               S(x) = [ S^1(x_1), S^2(x_1, x_2), ..., S^d(x_1, ..., x_d) ]^T

        2. Deep Composite Flow (num_steps = K > 1):
           Stacks K sequential autoregressive triangular SM-KAN blocks interleaved with
           alternating coordinate permutations (flips):
               z = S_K o P_{K-1} o S_{K-1} o ... o P_1 o S_1(x)
           This eliminates triangular ordering bias, allowing every variable to condition
           on every other variable while retaining exact analytic log-likelihood and fast
           1D Newton inversion.

    Exact closed-form log-density via lower-triangular Jacobian determinant:
        log pi_S(x) = - (d / 2) * log(2 * pi) - 0.5 * ||z||_2^2 + sum_{i=1}^K log |det nabla S_i|
    """

    def __init__(
        self,
        num_features: int,
        num_steps: int = 1,
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
            num_steps: Number of sequential triangular flow layers K (depth).
            hidden_dim: Hidden dimension Q (summation terms per KAN layer).
            grid_size: Number of B-spline grid intervals G (resolution).
            grid_range: (grid_min, grid_max) domain for B-spline knots.
            spline_order: Degree p of B-spline basis (3 for cubic splines).
            conditioning_scale: Whether to apply smooth scale modulation exp(sigma) * S_core + alpha.
            scale_bound: Maximum saturation bound s_max for scale modulation.
            penalty_type: Regularization penalty ('functional', 'coeff_group', or 'target_jacobian').
            dev: Device string ('cpu' or 'cuda').
        """
        super().__init__()
        self.num_features = num_features
        self.num_steps = num_steps
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.grid_range = grid_range
        self.spline_order = spline_order
        self.conditioning_scale = conditioning_scale
        self.scale_bound = float(scale_bound)
        self.penalty_type = penalty_type
        self.device = torch.device(dev)

        # Standardizing shift & scale buffers
        self.register_buffer("data_mean", torch.zeros(num_features, device=self.device))
        self.register_buffer("data_std", torch.ones(num_features, device=self.device))
        self.use_standardization = False

        blocks = []
        for _ in range(num_steps):
            block = SMKANDensityBlock(
                num_features=num_features,
                hidden_dim=hidden_dim,
                grid_size=grid_size,
                grid_range=grid_range,
                spline_order=spline_order,
                conditioning_scale=conditioning_scale,
                scale_bound=scale_bound,
                penalty_type=penalty_type,
                dev=dev,
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        # Backwards compatibility: expose maps of the first block
        self.maps = self.blocks[0].maps

    def count_parameters(self) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

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
        Pushforward mapping x -> z = S(x) through all K flow steps with exact log-det Jacobian.

        Args:
            x: Input tensor of shape (B, d).

        Returns:
            z: Latent standard normal vectors of shape (B, d).
            log_det_j: Total log-determinant log |det nabla S(x)| of shape (B,).
        """
        u, log_det_norm = self._normalize(x)
        cur = u
        log_det_j = log_det_norm.clone()

        for i, block in enumerate(self.blocks):
            if i > 0:
                # Permute / flip coordinate ordering between successive flow blocks
                cur = torch.flip(cur, dims=[-1])
            cur, ldj = block(cur)
            log_det_j = log_det_j + ldj

        return cur, log_det_j

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
        Computes regularized negative log-likelihood (NLL) loss over batch x:
            Loss = - (1/B) sum_{i=1}^B log pi_S(x^{(i)}) + reg_value * sum_k Penalty(S^k)
        """
        if self.num_steps == 1:
            # Fast single-pass evaluation for single-block models
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
        else:
            # Multi-step composite flow loss
            lp = self.log_prob(x)
            nll_loss = -lp.mean()
            total_reg = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            if reg_value > 0.0:
                for block in self.blocks:
                    for m in block.maps:
                        total_reg = total_reg + m.compute_regularization()
            total_loss = nll_loss + reg_value * total_reg
            return total_loss, nll_loss, total_reg

    def invert(
        self,
        z: torch.Tensor,
        max_iter: int = 30,
        tol: float = 1e-6,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Exact generative inversion: inverts sequential flow blocks in reverse order
        via batched 1D Newton-Raphson, then unnormalizes to original space.
        """
        dev = self.device if device is None else torch.device(device)
        cur = z.to(dev)
        all_diagnostics = []

        for i, block in enumerate(reversed(self.blocks)):
            cur, diags = block.invert(cur, max_iter=max_iter, tol=tol)
            all_diagnostics.extend(diags)
            if i < self.num_steps - 1:
                cur = torch.flip(cur, dims=[-1])

        x = self._unnormalize(cur)
        return x, all_diagnostics

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
            2. Solve S(u) = z backwards through all flow steps
            3. Return x = unnormalize(u)
        """
        dev = self.device if device is None else torch.device(device)
        z = torch.randn((n_samples, self.num_features), device=dev)
        x, _ = self.invert(z, max_iter=max_iter, tol=tol, device=dev)
        return x
