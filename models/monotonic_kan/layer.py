import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .basis import (
    BSplineBasis,
    BernsteinBasis,
    IntegratedRBFBasis,
    SOSChebyshevBasis,
)


class KANLayer(nn.Module):
    """
    A single Kolmogorov-Arnold Network (KAN) layer:
        x^{(l)}_j = sum_{i=1}^{n_in} phi_{j, i}(x^{(l-1)}_i) + bias_j

    Where each univariate edge activation phi_{j, i}: R -> R is parameterized
    by one of four functional bases:
        1. 'bspline': Uniform cubic B-splines (piecewise Cox-de Boor recursion).
        2. 'bernstein': Knot-free Bernstein polynomials with precomputed matrix derivatives.
        3. 'rbf' / 'integrated_rbf': Integrated Gaussian Radial Basis Functions via closed-form erf.
        4. 'chebyshev' / 'sos_chebyshev': Sum-of-Squares Chebyshev polynomials via Cholesky factor.

    Supports arbitrary input and output dimensions, and per-edge monotonic
    constraints via an optional `monotonic_mask` of shape (out_dim, in_dim).
    When monotonic_mask[j, i] is True, the edge activation is constrained to be strictly
    increasing, guaranteeing phi'_{j, i}(u) >= eps_mono > 0 everywhere.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        grid_size: int = 8,
        grid_range: tuple[float, float] = (-4.0, 4.0),
        spline_order: int = 3,
        monotonic_mask: torch.Tensor | None = None,
        eps_mono: float = 1e-4,
        bias: bool = True,
        basis_type: str = "bspline",
        degree: int | None = None,
        dev: str | torch.device = "cpu",
    ):
        """
        Args:
            in_dim: Number of input neurons n_{l-1}.
            out_dim: Number of output neurons n_l.
            grid_size: Number of intervals in B-spline knot grid (or default degree/centers).
            grid_range: (grid_min, grid_max) for the standardized domain.
            spline_order: Spline degree (default 3 for cubic splines).
            monotonic_mask: Boolean tensor of shape (out_dim, in_dim).
                            True marks edges that must be monotonically increasing.
            eps_mono: Strict positive floor added to softplus increments.
            bias: Whether to include an additive bias per output neuron.
            basis_type: 'bspline', 'bernstein', 'rbf', or 'chebyshev'.
            degree: Optional explicit degree or center count.
            dev: Target PyTorch device.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.grid_range = grid_range
        self.spline_order = spline_order
        self.eps_mono = eps_mono
        self.basis_type = str(basis_type).lower()
        self.device = torch.device(dev)

        if self.basis_type == "bernstein":
            self.degree = (
                degree
                if degree is not None
                else (spline_order if spline_order > 3 else grid_size)
            )
            grid = BernsteinBasis.build_grid(
                grid_min=grid_range[0],
                grid_max=grid_range[1],
                degree=self.degree,
                device=self.device,
            )
            self.register_buffer("grid", grid)
            self.n_bases = self.degree + 1
            greville = BernsteinBasis.get_greville_abscissae(grid, self.degree)
            self.register_buffer("greville", greville)
            self.step_h = (grid_range[1] - grid_range[0]) / float(self.degree)

        elif self.basis_type in ["rbf", "integrated_rbf"]:
            self.degree = degree if degree is not None else grid_size
            grid = IntegratedRBFBasis.build_grid(
                grid_min=grid_range[0],
                grid_max=grid_range[1],
                degree=self.degree,
                device=self.device,
            )
            self.register_buffer("grid", grid)
            # n_bases: 1 linear carrier term + self.degree RBF centers
            self.n_bases = self.degree + 1
            greville = IntegratedRBFBasis.get_greville_abscissae(grid, self.degree)
            self.register_buffer("greville", greville)
            self.step_h = (grid_range[1] - grid_range[0]) / float(self.degree)

        elif self.basis_type in ["chebyshev", "sos_chebyshev"]:
            self.degree = (
                degree
                if degree is not None
                else (spline_order if spline_order > 3 else grid_size)
            )
            self.K = max(1, self.degree // 2)
            self.N = 2 * self.K
            self.n_bases = self.N + 1
            grid = SOSChebyshevBasis.build_grid(
                grid_min=grid_range[0],
                grid_max=grid_range[1],
                degree=self.degree,
                device=self.device,
            )
            self.register_buffer("grid", grid)
            greville = SOSChebyshevBasis.get_greville_abscissae(grid, self.degree)
            self.register_buffer("greville", greville)
            self.step_h = (grid_range[1] - grid_range[0]) / float(self.N)

            # Precomputed contraction tensor S of shape (N+1, K+1, K+1)
            S = torch.zeros(self.N + 1, self.K + 1, self.K + 1, device=self.device)
            for j in range(self.K + 1):
                for k in range(self.K + 1):
                    n1 = j + k
                    n2 = abs(j - k)
                    S[n1, j, k] += 0.5
                    S[n2, j, k] += 0.5
            self.register_buffer("cheb_S", S)

        elif self.basis_type == "bspline":
            self.degree = spline_order
            grid = BSplineBasis.build_grid(
                grid_min=grid_range[0],
                grid_max=grid_range[1],
                num_intervals=grid_size,
                degree=spline_order,
                device=self.device,
            )
            self.register_buffer("grid", grid)
            self.n_bases = len(grid) - spline_order - 1
            greville = BSplineBasis.get_greville_abscissae(grid, spline_order)
            self.register_buffer("greville", greville)
            self.step_h = (grid_range[1] - grid_range[0]) / float(grid_size)
        else:
            raise ValueError(
                f"Unsupported basis_type: {self.basis_type}. Expected 'bspline', 'bernstein', 'rbf', or 'chebyshev'."
            )

        # Register monotonic edge mask
        if monotonic_mask is not None:
            self.register_buffer("monotonic_mask", monotonic_mask.to(self.device).bool())
        else:
            self.register_buffer(
                "monotonic_mask", torch.zeros(out_dim, in_dim, device=self.device, dtype=torch.bool)
            )

        self.has_monotonic = bool(self.monotonic_mask.any().item())
        self.has_unconstrained = bool((~self.monotonic_mask).any().item())

        # Parameter tensors
        if self.has_monotonic:
            if self.basis_type in ["chebyshev", "sos_chebyshev"]:
                self.L_param = nn.Parameter(
                    torch.randn(out_dim, in_dim, self.K + 1, self.K + 1, device=self.device) * 0.05
                )
                self.c0 = None
                self.delta = None
            elif self.basis_type in ["rbf", "integrated_rbf"]:
                # delta[..., 0] is linear term, delta[..., 1:] are RBF weights
                self.delta = nn.Parameter(
                    torch.zeros(out_dim, in_dim, self.n_bases, device=self.device)
                )
                self.c0 = nn.Parameter(torch.zeros(out_dim, in_dim, 1, device=self.device))
                self.L_param = None
            else:  # bspline and bernstein
                self.c0 = nn.Parameter(torch.zeros(out_dim, in_dim, 1, device=self.device))
                self.delta = nn.Parameter(
                    torch.zeros(out_dim, in_dim, self.n_bases - 1, device=self.device)
                )
                self.L_param = None
        else:
            self.c0 = None
            self.delta = None
            self.L_param = None

        if self.has_unconstrained:
            self.unconstrained_weights = nn.Parameter(
                torch.randn(out_dim, in_dim, self.n_bases, device=self.device) * 0.01
            )
        else:
            self.unconstrained_weights = None

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim, device=self.device))
        else:
            self.bias = None

    def get_coeffs(self) -> torch.Tensor:
        """
        Assembles control points for all (out_dim, in_dim) edges into a unified
        tensor of shape (out_dim, in_dim, n_bases).
        """
        if self.has_monotonic:
            if self.basis_type in ["chebyshev", "sos_chebyshev"]:
                L = torch.tril(self.L_param)
                Q = torch.matmul(L, L.transpose(-1, -2))
                c = torch.einsum("njk,oijk->oin", self.cheb_S, Q)
                mono_coeffs = c.clone()
                mono_coeffs[..., 0] = mono_coeffs[..., 0] + self.eps_mono
            elif self.basis_type in ["rbf", "integrated_rbf"]:
                w_lin = F.softplus(self.delta[..., :1]) + self.eps_mono
                w_rbf = F.softplus(self.delta[..., 1:])
                mono_coeffs = torch.cat([w_lin, w_rbf], dim=-1)
            else:  # bspline and bernstein
                diffs = F.softplus(self.delta) + self.eps_mono
                mono_coeffs = torch.cat([self.c0, self.c0 + torch.cumsum(diffs, dim=-1)], dim=-1)

            if self.has_unconstrained:
                return torch.where(
                    self.monotonic_mask.unsqueeze(-1), mono_coeffs, self.unconstrained_weights
                )
            return mono_coeffs
        else:
            return self.unconstrained_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward evaluation: x^{(l)}_j = sum_i phi_{j, i}(x^{(l-1)}_i) + bias_j.
        """
        if self.basis_type == "bernstein":
            b_val, _, _, _ = BernsteinBasis.eval_basis(x, self.grid, self.degree)
        elif self.basis_type in ["rbf", "integrated_rbf"]:
            b_val, _, _, _ = IntegratedRBFBasis.eval_basis(x, self.grid, self.degree)
        elif self.basis_type in ["chebyshev", "sos_chebyshev"]:
            b_val, _, _, _ = SOSChebyshevBasis.eval_basis(x, self.grid, self.degree)
        else:
            b_val, _, _, _ = BSplineBasis.eval_basis(x, self.grid, self.spline_order)

        coeffs = self.get_coeffs()
        out = torch.einsum("bim,jim->bj", b_val, coeffs)
        if self.has_monotonic and self.c0 is not None and self.basis_type in ["rbf", "integrated_rbf"]:
            out = out + self.c0.squeeze(-1).sum(dim=-1)
        if self.bias is not None:
            out = out + self.bias
        return out

    def eval_splines_and_derivatives(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluates layer activations and all local 1st, 2nd, and 3rd derivatives.
        """
        if self.basis_type == "bernstein":
            b_val, b_d1, b_d2, b_d3 = BernsteinBasis.eval_basis(x, self.grid, self.degree)
        elif self.basis_type in ["rbf", "integrated_rbf"]:
            b_val, b_d1, b_d2, b_d3 = IntegratedRBFBasis.eval_basis(x, self.grid, self.degree)
        elif self.basis_type in ["chebyshev", "sos_chebyshev"]:
            b_val, b_d1, b_d2, b_d3 = SOSChebyshevBasis.eval_basis(x, self.grid, self.degree)
        else:
            b_val, b_d1, b_d2, b_d3 = BSplineBasis.eval_basis(x, self.grid, self.spline_order)

        coeffs = self.get_coeffs()

        out = torch.einsum("bim,jim->bj", b_val, coeffs)
        if self.has_monotonic and self.c0 is not None and self.basis_type in ["rbf", "integrated_rbf"]:
            out = out + self.c0.squeeze(-1).sum(dim=-1)
        if self.bias is not None:
            out = out + self.bias

        phi_prime = torch.einsum("bim,jim->bji", b_d1, coeffs)
        phi_pprime = torch.einsum("bim,jim->bji", b_d2, coeffs)
        phi_ppprime = torch.einsum("bim,jim->bji", b_d3, coeffs)

        return out, phi_prime, phi_pprime, phi_ppprime

    def initialize_identity(self, active_edge: tuple[int, int] | None = None):
        """
        Initializes the layer. If active_edge=(out_idx, in_idx) is specified:
            phi_{out_idx, in_idx}(u) approx u.
        """
        with torch.no_grad():
            if self.basis_type in ["rbf", "integrated_rbf"]:
                if self.has_monotonic:
                    init_lin = math.log(max(math.expm1(1.0 - self.eps_mono), 1e-4))
                    self.delta.fill_(-12.0)
                    if self.c0 is not None:
                        self.c0.fill_(0.0)
                    if active_edge is not None:
                        out_idx, in_idx = active_edge
                        self.delta[out_idx, in_idx, 0].fill_(init_lin)
            elif self.basis_type in ["chebyshev", "sos_chebyshev"]:
                if self.has_monotonic:
                    self.L_param.zero_()
                    if active_edge is not None:
                        out_idx, in_idx = active_edge
                        self.L_param[out_idx, in_idx, 0, 0] = math.sqrt(max(1.0 - self.eps_mono, 1e-4))
            else:  # bspline and bernstein
                init_delta_active = torch.log(
                    torch.expm1(torch.tensor(max(self.step_h - self.eps_mono, 1e-4)))
                )
                if self.has_monotonic:
                    self.delta.fill_(-12.0)
                    self.c0.fill_(0.0)
                    if active_edge is not None:
                        out_idx, in_idx = active_edge
                        self.delta[out_idx, in_idx].fill_(init_delta_active.item())
                        self.c0[out_idx, in_idx].fill_(self.greville[0].item())

            if self.has_unconstrained:
                self.unconstrained_weights.zero_()

            if self.bias is not None:
                self.bias.zero_()


class DeepKAN(nn.Module):
    """
    Arbitrary-depth unconstrained Kolmogorov-Arnold Network:
        [n_0 -> n_1 -> n_2 -> ... -> n_L]

    Provides exact programmatic Einstein summation evaluation for forward pass
    and input Jacobian matrix J = dx^{(L)} / dx^{(0)}.
    """

    def __init__(
        self,
        layers_hidden: list[int] | tuple[int, ...],
        grid_size: int = 8,
        grid_range: tuple[float, float] = (-4.0, 4.0),
        spline_order: int = 3,
        bias: bool = True,
        basis_type: str = "bspline",
        degree: int | None = None,
        dev: str | torch.device = "cpu",
    ):
        super().__init__()
        self.layers_hidden = list(layers_hidden)
        self.basis_type = basis_type
        self.degree = degree
        self.device = torch.device(dev)

        self.layers = nn.ModuleList([
            KANLayer(
                in_dim=self.layers_hidden[l],
                out_dim=self.layers_hidden[l + 1],
                grid_size=grid_size,
                grid_range=grid_range,
                spline_order=spline_order,
                monotonic_mask=None,
                bias=bias,
                basis_type=basis_type,
                degree=degree,
                dev=self.device,
            )
            for l in range(len(self.layers_hidden) - 1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acts = x
        for layer in self.layers:
            acts = layer(acts)
        return acts

    def compute_forward_and_jacobian(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B_size, d_in = x.shape
        acts = x
        A = torch.eye(d_in, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B_size, d_in, d_in)

        for layer in self.layers:
            acts, p1, _, _ = layer.eval_splines_and_derivatives(acts)
            A = torch.einsum("bai,bij->baj", p1, A)

        if self.layers_hidden[-1] == 1:
            return acts, A.squeeze(1)
        return acts, A

    def initialize_zeros(self):
        with torch.no_grad():
            for layer in self.layers:
                layer.initialize_identity(active_edge=None)
