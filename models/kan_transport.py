import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BSplineBasis:
    """
    Evaluates uniform B-spline basis functions and their analytic derivatives
    up to degree 3, with continuous linear tail extrapolation beyond [grid_min, grid_max].
    Provides C^2 continuity across knots, ideal for 3rd-order mixed derivatives.
    """

    @staticmethod
    def build_grid(grid_min=-4.0, grid_max=4.0, num_intervals=8, degree=3, device="cpu", dtype=torch.float32):
        """
        Builds uniform knot grid padded with degree knots on left and right.
        Total knots: num_intervals + 1 + 2 * degree.
        Number of basis functions: num_intervals + degree.
        """
        step = (grid_max - grid_min) / float(num_intervals)
        inner = torch.linspace(grid_min, grid_max, num_intervals + 1, device=device, dtype=dtype)
        left = inner[0] - step * torch.arange(degree, 0, -1, device=device, dtype=dtype)
        right = inner[-1] + step * torch.arange(1, degree + 1, device=device, dtype=dtype)
        return torch.cat([left, inner, right])

    @staticmethod
    def get_greville_abscissae(grid, degree=3):
        """
        Computes Greville abscissae (knot averages) for degree-p B-spline basis functions.
        For basis i (i = 0 ... n_bases - 1):
            xi_i = sum(grid[i + 1 : i + degree + 1]) / degree.
        Marsden's identity guarantees sum_i xi_i * B_i(x) = x for all x in [grid_min, grid_max].
        """
        n_bases = len(grid) - degree - 1
        greville = torch.zeros(n_bases, device=grid.device, dtype=grid.dtype)
        for i in range(n_bases):
            greville[i] = grid[i + 1 : i + degree + 1].sum() / float(degree)
        return greville

    @staticmethod
    def eval_basis(x, grid, degree=3):
        """
        Vectorized evaluation of B-spline basis functions and 1st, 2nd, and 3rd derivatives.

        Args:
            x: Input tensor of shape (..., )
            grid: Knot grid tensor of shape (num_knots,)
            degree: Spline degree (typically 3 for cubic splines)

        Returns:
            b_val: Basis values of shape (..., n_bases)
            b_d1: 1st derivatives of shape (..., n_bases)
            b_d2: 2nd derivatives of shape (..., n_bases)
            b_d3: 3rd derivatives of shape (..., n_bases)
        """
        t_min = grid[degree]
        t_max = grid[-degree - 1]
        eps_boundary = 1e-6

        def _eval_interior(u):
            u_in = u.unsqueeze(-1)
            bases = [((u_in >= grid[:-1]) & (u_in < grid[1:])).to(u.dtype)]
            for p in range(1, degree + 1):
                denom1 = grid[p:-1] - grid[: -(p + 1)]
                denom2 = grid[(p + 1) :] - grid[1:-p]
                term1 = torch.where(
                    denom1 > 0,
                    (u_in - grid[: -(p + 1)]) / denom1 * bases[-1][..., :-1],
                    torch.zeros_like(u_in),
                )
                term2 = torch.where(
                    denom2 > 0,
                    (grid[(p + 1) :] - u_in) / denom2 * bases[-1][..., 1:],
                    torch.zeros_like(u_in),
                )
                bases.append(term1 + term2)

            b_curr = bases[3]

            # 1st derivative of cubic spline (degree 3)
            d1_term = bases[2] / (grid[3:] - grid[:-3])
            b_d1 = 3.0 * (d1_term[..., :-1] - d1_term[..., 1:])

            # 2nd derivative of cubic spline
            d2_term = bases[1] / (grid[2:] - grid[:-2])
            b2_d1 = 2.0 * (d2_term[..., :-1] - d2_term[..., 1:])
            d2_scaled = b2_d1 / (grid[3:] - grid[:-3])
            b_d2 = 3.0 * (d2_scaled[..., :-1] - d2_scaled[..., 1:])

            # 3rd derivative of cubic spline
            d3_term = bases[0] / (grid[1:] - grid[:-1])
            b1_d1 = 1.0 * (d3_term[..., :-1] - d3_term[..., 1:])
            b2_d2 = 2.0 * ((b1_d1 / (grid[2:] - grid[:-2]))[..., :-1] - (b1_d1 / (grid[2:] - grid[:-2]))[..., 1:])
            d3_scaled = b2_d2 / (grid[3:] - grid[:-3])
            b_d3 = 3.0 * (d3_scaled[..., :-1] - d3_scaled[..., 1:])

            return b_curr, b_d1, b_d2, b_d3

        # Boundary anchor values for linear tail extrapolation
        b_min, db_min, _, _ = _eval_interior(t_min.unsqueeze(0))
        b_max, db_max, _, _ = _eval_interior((t_max - eps_boundary).unsqueeze(0))

        # Evaluate clamped interior
        x_clamped = torch.clamp(x, t_min, t_max - eps_boundary)
        b_curr, b_d1, b_d2, b_d3 = _eval_interior(x_clamped)

        # Extrapolation masks
        is_left = (x < t_min).unsqueeze(-1)
        is_right = (x >= t_max).unsqueeze(-1)
        dx_left = (x - t_min).unsqueeze(-1)
        dx_right = (x - t_max).unsqueeze(-1)

        b_val = torch.where(
            is_left,
            b_min + db_min * dx_left,
            torch.where(is_right, b_max + db_max * dx_right, b_curr),
        )
        b_d1 = torch.where(is_left, db_min, torch.where(is_right, db_max, b_d1))
        b_d2 = torch.where(is_left | is_right, torch.zeros_like(b_d2), b_d2)
        b_d3 = torch.where(is_left | is_right, torch.zeros_like(b_d3), b_d3)

        return b_val, b_d1, b_d2, b_d3


class CubicHermiteBasis:
    """
    Alternative spline engine using Cubic Hermite Splines (PCHIP), inspired by MonoKAN (Polo-Molina et al., 2025).
    Evaluates explicit algebraic degree-3 Hermite polynomials:
        h00(t) = 2t^3 - 3t^2 + 1,  h10(t) = t^3 - 2t^2 + t
        h01(t) = -2t^3 + 3t^2,     h11(t) = t^3 - t^2
    with linear tail extrapolation using endpoint slopes. Provides C^1 continuity.
    """

    @staticmethod
    def eval_hermite(x, grid):
        """
        Evaluates Cubic Hermite basis functions and 1st derivatives.

        Args:
            x: Input tensor of shape (...,)
            grid: Uniform knot grid of shape (num_knots,)

        Returns:
            basis_vals: Tuple (h00, h10, h01, h11)
            basis_d1: Tuple (dh00, dh10, dh01, dh11) wrt x
            interval_idx: Tensor of interval indices k
        """
        num_intervals = len(grid) - 1
        t_min = grid[0]
        t_max = grid[-1]
        h = (t_max - t_min) / float(num_intervals)
        inv_h = 1.0 / h

        x_clamped = torch.clamp(x, t_min, t_max - 1e-6)
        k = torch.clamp(((x_clamped - t_min) / h).long(), 0, num_intervals - 1)
        xk = grid[k]
        t = (x_clamped - xk) / h

        t2 = t * t
        t3 = t2 * t

        h00 = 2.0 * t3 - 3.0 * t2 + 1.0
        h10 = t3 - 2.0 * t2 + t
        h01 = -2.0 * t3 + 3.0 * t2
        h11 = t3 - t2

        dh00 = (6.0 * t2 - 6.0 * t) * inv_h
        dh10 = (3.0 * t2 - 4.0 * t + 1.0)
        dh01 = (-6.0 * t2 + 6.0 * t) * inv_h
        dh11 = (3.0 * t2 - 2.0 * t)

        return (h00, h10, h01, h11), (dh00, dh10, dh01, dh11), k


class MonotonicKANMap(nn.Module):
    """
    Modular Monotonic Kolmogorov-Arnold Network (KAN) Transport Map Component S^k(x):

        Standard Additive Mode (conditioning_scale=False):
            S^k(x) = sum_{q=1}^Q Phi_{2, q}( phi_{1, q, k}(x_k) + sum_{j != k} phi_{1, q, j}(x_j) ) + beta

        Scale-Modulated Mode (conditioning_scale=True):
            S^k(x) = exp( sigma(x_{-k}) ) * S^k_{core}(x) + alpha(x_{-k})

    Guarantees strict monotonicity along target dimension x_k (partial_k S^k > 0) everywhere:
        partial_k S^k(x) = exp(sigma(x_{-k})) * sum_{q=1}^Q Phi'_{2, q}(u_q(x)) * phi'_{1, q, k}(x_k) > 0

    Eliminates numerical quadrature via closed-form analytic forward, derivatives, and
    generalized precision entries.
    """

    def __init__(
        self,
        num_features,
        target_index=0,
        hidden_dim=4,
        grid_size=8,
        grid_range=(-4.0, 4.0),
        spline_order=3,
        penalty_type="functional",
        conditioning_scale=False,
        scale_bound=4.0,
        dev="cpu",
    ):
        """
        Args:
            num_features (int): Dimensionality d of the data.
            target_index (int): Index k of the target feature being transported.
            hidden_dim (int): Number of summation terms Q in KAN layer.
            grid_size (int): Number of intervals in B-spline grid.
            grid_range (tuple): (grid_min, grid_max) for the standardized domain.
            spline_order (int): Spline degree p (default 3 for cubic splines).
            penalty_type (str): Regularization penalty: 'functional', 'coeff_group', or 'target_jacobian'.
            conditioning_scale (bool): Whether to enable scale modulation exp(sigma(x_{-k})) * S_core + alpha.
            scale_bound (float): Saturation bound for smooth tanh scale modulation: sigma = scale_bound * tanh(sigma_raw / scale_bound).
            dev (str): Target PyTorch device.
        """
        super(MonotonicKANMap, self).__init__()
        self.num_features = num_features
        self.target_index = target_index
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.grid_range = grid_range
        self.spline_order = spline_order
        self.penalty_type = penalty_type
        self.conditioning_scale = conditioning_scale
        self.scale_bound = float(scale_bound)
        self.device = torch.device(dev)
        self.other_indices = [j for j in range(num_features) if j != target_index]

        # Register knot grids as persistent buffers
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

        # Step size h between Greville points
        step_h = (grid_range[1] - grid_range[0]) / float(grid_size)
        self.step_h = step_h
        self.eps_mono = 1e-4

        # -------------------------------------------------------------
        # 1. Inner Layer Parameters:
        #    Conditioning features (j != k): unconstrained spline weights
        #    Target feature (j = k): monotonic increasing weights
        # -------------------------------------------------------------
        if self.num_features > 1:
            self.inner_cond_weights = nn.Parameter(
                torch.randn(self.num_features - 1, self.hidden_dim, self.n_bases, device=self.device) * 0.01
            )
        else:
            self.inner_cond_weights = None

        # Monotonic target parameters
        self.inner_target_c0 = nn.Parameter(torch.zeros(self.hidden_dim, 1, device=self.device))
        self.inner_target_delta = nn.Parameter(
            torch.zeros(self.hidden_dim, self.n_bases - 1, device=self.device)
        )

        # -------------------------------------------------------------
        # 2. Outer Layer Parameters:
        #    Monotonic increasing weights for Phi_{2, q}
        # -------------------------------------------------------------
        self.outer_c0 = nn.Parameter(torch.zeros(self.hidden_dim, 1, device=self.device))
        self.outer_delta = nn.Parameter(
            torch.zeros(self.hidden_dim, self.n_bases - 1, device=self.device)
        )

        # Bias term
        self.bias = nn.Parameter(torch.zeros(1, device=self.device))

        # -------------------------------------------------------------
        # 3. Conditioning Scale Modulation (Optional)
        #    sigma(x_{-k}) = sum_{j != k} psi_j(x_j) + b_sigma
        #    alpha(x_{-k}) = sum_{j != k} rho_j(x_j) + b_alpha
        # -------------------------------------------------------------
        if self.conditioning_scale and len(self.other_indices) > 0:
            self.scale_weights = nn.Parameter(
                torch.zeros(len(self.other_indices), self.n_bases, device=self.device)
            )
            self.scale_bias = nn.Parameter(torch.zeros(1, device=self.device))
            self.offset_weights = nn.Parameter(
                torch.zeros(len(self.other_indices), self.n_bases, device=self.device)
            )
            self.offset_bias = nn.Parameter(torch.zeros(1, device=self.device))
        else:
            self.scale_weights = None
            self.scale_bias = None
            self.offset_weights = None
            self.offset_bias = None

        # Initialize to Identity Map: S^k(x) \approx x_k
        self._initialize_identity()

    def _initialize_identity(self):
        """
        Initializes control coefficients such that:
            phi_{1, q, k}(x_k) = (1/Q) * x_k
            Phi_{2, q}(u) = u
            S^k(x) = sum_q Phi_{2, q}( (1/Q) x_k ) = x_k
            partial_k S^k(x) = 1.0
        """
        with torch.no_grad():
            # Target inner step: h / Q
            target_step = self.step_h / float(self.hidden_dim)
            if target_step > self.eps_mono:
                init_delta_inner = torch.log(torch.expm1(torch.tensor(target_step - self.eps_mono)))
            else:
                init_delta_inner = torch.tensor(-5.0)
            self.inner_target_delta.fill_(init_delta_inner.item())

            # Target c0 = greville[0] / Q
            self.inner_target_c0.fill_((self.greville[0] / float(self.hidden_dim)).item())

            # Outer step: h
            outer_step = self.step_h
            if outer_step > self.eps_mono:
                init_delta_outer = torch.log(torch.expm1(torch.tensor(outer_step - self.eps_mono)))
            else:
                init_delta_outer = torch.tensor(-5.0)
            self.outer_delta.fill_(init_delta_outer.item())

            # Outer c0 = greville[0]
            self.outer_c0.fill_(self.greville[0].item())

            # Conditioning weights initialized to zero / near zero
            if self.inner_cond_weights is not None:
                self.inner_cond_weights.zero_()

            self.bias.zero_()

            if self.scale_weights is not None:
                self.scale_weights.zero_()
                self.scale_bias.zero_()
                self.offset_weights.zero_()
                self.offset_bias.zero_()

    def get_inner_target_coeffs(self):
        """
        Returns strictly monotonic inner coefficients for target variable x_k.
        Shape: (Q, n_bases), strictly increasing across n_bases dimension.
        """
        diffs = F.softplus(self.inner_target_delta) + self.eps_mono
        return torch.cat([self.inner_target_c0, self.inner_target_c0 + torch.cumsum(diffs, dim=-1)], dim=-1)

    def get_outer_coeffs(self):
        """
        Returns strictly monotonic outer coefficients for Phi_{2, q}.
        Shape: (Q, n_bases), strictly increasing across n_bases dimension.
        """
        diffs = F.softplus(self.outer_delta) + self.eps_mono
        return torch.cat([self.outer_c0, self.outer_c0 + torch.cumsum(diffs, dim=-1)], dim=-1)

    def get_full_inner_coeffs(self):
        """
        Assembles all inner coefficients into a single tensor of shape (d, Q, n_bases).
        """
        d = self.num_features
        Q = self.hidden_dim
        M = self.n_bases

        target_coeffs = self.get_inner_target_coeffs()  # (Q, M)

        if d == 1:
            return target_coeffs.unsqueeze(0)  # (1, Q, M)

        inner_coeffs = torch.empty(d, Q, M, device=self.device, dtype=target_coeffs.dtype)
        inner_coeffs[self.target_index] = target_coeffs
        for idx, j in enumerate(self.other_indices):
            inner_coeffs[j] = self.inner_cond_weights[idx]

        return inner_coeffs

    def _assemble_input(self, target_data, non_target_data=None):
        if non_target_data is None:
            if target_data.dim() == 1:
                target_data = target_data.unsqueeze(0)
            return target_data.to(self.device)

        B = target_data.shape[0]
        x = torch.empty(B, self.num_features, device=self.device, dtype=target_data.dtype)
        x[:, [self.target_index]] = target_data.to(self.device)
        if len(self.other_indices) > 0:
            x[:, self.other_indices] = non_target_data.to(self.device)
        return x

    def forward(self, target_data, non_target_data=None):
        """
        Forward evaluation S^k(x). Adheres to UMNN interface:
            forward(target_data, non_target_data) -> (B, 1)
        """
        x = self._assemble_input(target_data, non_target_data)

        # Evaluate B-spline basis functions for all features
        b_val, _, _, _ = BSplineBasis.eval_basis(x, self.grid, self.spline_order)  # (B, d, n_bases)

        inner_coeffs = self.get_full_inner_coeffs()  # (d, Q, n_bases)
        u = torch.einsum("bdm,dqm->bq", b_val, inner_coeffs)  # (B, Q)

        outer_coeffs = self.get_outer_coeffs()  # (Q, n_bases)
        Phi_val, _, _, _ = BSplineBasis.eval_basis(u, self.grid, self.spline_order)  # (B, Q, n_bases)

        out_core = torch.einsum("bqm,qm->b", Phi_val, outer_coeffs) + self.bias

        if self.scale_weights is not None:
            b_val_cond = b_val[:, self.other_indices, :]
            sigma_raw = torch.einsum("bdm,dm->b", b_val_cond, self.scale_weights) + self.scale_bias
            sigma = self.scale_bound * torch.tanh(sigma_raw / self.scale_bound)
            E = torch.exp(sigma)
            offset = torch.einsum("bdm,dm->b", b_val_cond, self.offset_weights) + self.offset_bias
            out = E * out_core + offset
        else:
            out = out_core

        return out.unsqueeze(1)

    def compute_derivatives(self, target_data, non_target_data=None):
        """
        Closed-form analytic computation of forward pass and all necessary derivatives:
            - S: S^k(x) of shape (B, 1)
            - S_k: partial_k S^k(x) > 0 of shape (B, 1)
            - S_all: partial_j S^k(x) for all j of shape (B, d)
            - S_jk: partial_j partial_k S^k(x) for all j of shape (B, d)
            - S_kk: partial_k^2 S^k(x) of shape (B, 1)
            - S_jkk: partial_j partial_k^2 S^k(x) for all j of shape (B, d)
        """
        x = self._assemble_input(target_data, non_target_data)

        b_val, b_d1, b_d2, b_d3 = BSplineBasis.eval_basis(x, self.grid, self.spline_order)
        inner_coeffs = self.get_full_inner_coeffs()

        u = torch.einsum("bdm,dqm->bq", b_val, inner_coeffs)
        phi_prime = torch.einsum("bdm,dqm->bdq", b_d1, inner_coeffs)
        phi_pprime = torch.einsum("bdm,dqm->bdq", b_d2, inner_coeffs)

        outer_coeffs = self.get_outer_coeffs()
        Phi_val, Phi_d1, Phi_d2, Phi_d3 = BSplineBasis.eval_basis(u, self.grid, self.spline_order)

        # Output S^k_core(x)
        out_core = (torch.einsum("bqm,qm->b", Phi_val, outer_coeffs) + self.bias).unsqueeze(1)

        # Outer derivatives
        Phi_p = torch.einsum("bqm,qm->bq", Phi_d1, outer_coeffs)
        Phi_pp = torch.einsum("bqm,qm->bq", Phi_d2, outer_coeffs)
        Phi_ppp = torch.einsum("bqm,qm->bq", Phi_d3, outer_coeffs)

        # All 1st derivatives core: partial_j S^k_core(x) = sum_q Phi'_q * phi'_{j, q}
        S_all_core = torch.einsum("bq,bdq->bd", Phi_p, phi_prime)
        S_k_core = S_all_core[:, [self.target_index]]

        # Target 1st derivative phi'_{k, q}
        phi_k_p = phi_prime[:, self.target_index, :]  # (B, Q)
        phi_k_pp = phi_pprime[:, self.target_index, :]  # (B, Q)

        # Mixed 2nd derivatives core: partial_j partial_k S^k_core(x) = sum_q Phi''_q * phi'_{j, q} * phi'_{k, q}
        S_jk_core = torch.einsum("bq,bdq,bq->bd", Phi_pp, phi_prime, phi_k_p)

        # Target 2nd derivative core: partial_k^2 S^k_core(x)
        S_kk_core = (
            torch.einsum("bq,bq->b", Phi_pp, phi_k_p**2)
            + torch.einsum("bq,bq->b", Phi_p, phi_k_pp)
        ).unsqueeze(1)

        # Mixed 3rd derivatives core: partial_j partial_k^2 S^k_core(x)
        term_inner = (
            torch.einsum("bq,bq->bq", Phi_ppp, phi_k_p**2)
            + torch.einsum("bq,bq->bq", Phi_pp, phi_k_pp)
        )
        S_jkk_core = torch.einsum("bdq,bq->bd", phi_prime, term_inner)

        # Apply Conditioning Scale Modulation if enabled
        if self.scale_weights is not None and len(self.other_indices) > 0:
            b_val_cond = b_val[:, self.other_indices, :]
            b_d1_cond = b_d1[:, self.other_indices, :]

            # sigma(x_{-k}) and E = exp(sigma) via smooth tanh saturation
            sigma_raw = torch.einsum("bdm,dm->b", b_val_cond, self.scale_weights) + self.scale_bias
            tanh_val = torch.tanh(sigma_raw / self.scale_bound)
            sigma = self.scale_bound * tanh_val
            E = torch.exp(sigma).unsqueeze(1)  # (B, 1)

            # sigma derivatives wrt x: shape (B, d)
            # By chain rule: d(sigma)/dx_j = (1 - tanh^2(sigma_raw / scale_bound)) * d(sigma_raw)/dx_j
            sech2 = (1.0 - tanh_val**2).unsqueeze(1)
            sig_d1_raw = torch.einsum("bdm,dm->bd", b_d1_cond, self.scale_weights)
            sig_d1 = sech2 * sig_d1_raw
            sigma_all = torch.zeros(x.shape[0], self.num_features, device=x.device, dtype=x.dtype)
            sigma_all[:, self.other_indices] = sig_d1
            # alpha(x_{-k}) and alpha derivatives: shape (B, 1) and (B, d)
            offset = torch.einsum("bdm,dm->b", b_val_cond, self.offset_weights) + self.offset_bias
            alpha = offset.unsqueeze(1)
            alp_d1 = torch.einsum("bdm,dm->bd", b_d1_cond, self.offset_weights)
            alpha_all = torch.zeros(x.shape[0], self.num_features, device=x.device, dtype=x.dtype)
            alpha_all[:, self.other_indices] = alp_d1

            # Composed outputs:
            # S = E * S_core + alpha
            out = E * out_core + alpha
            # S_k = E * S_k_core
            S_k = E * S_k_core
            # S_j = E * (sigma_j * S_core + S_j_core) + alpha_j
            S_all = E * (sigma_all * out_core + S_all_core) + alpha_all
            S_all[:, [self.target_index]] = S_k

            # S_jk = E * (sigma_j * S_k_core + S_jk_core)
            S_jk = E * (sigma_all * S_k_core + S_jk_core)

            # S_kk = E * S_kk_core
            S_kk = E * S_kk_core

            # S_jkk = E * (sigma_j * S_kk_core + S_jkk_core)
            S_jkk = E * (sigma_all * S_kk_core + S_jkk_core)
        else:
            out = out_core
            S_k = S_k_core
            S_all = S_all_core
            S_jk = S_jk_core
            S_kk = S_kk_core
            S_jkk = S_jkk_core

        return {
            "out": out,
            "S_k": S_k,
            "S_all": S_all,
            "S_jk": S_jk,
            "S_kk": S_kk,
            "S_jkk": S_jkk,
        }

    def compute_regularization(self, S_all=None):
        """
        Computes regularization penalty:
            1. 'functional' (L-SING baseline Eq 5): sum_{j} sqrt( (1/M) sum_i (partial_j S^k(x^i))^2 )
            2. 'coeff_group' (KAN native): sum_{j != k} sqrt( sum_q sum_m |c_{q, j, m}|^2 )
            3. 'target_jacobian': sqrt( (partial_k S^k)^2.mean() )
        """
        eps = 1e-8
        if self.penalty_type == "coeff_group":
            if self.inner_cond_weights is None:
                return torch.tensor(0.0, device=self.device)
            feature_norms = torch.sqrt(torch.sum(self.inner_cond_weights**2, dim=(1, 2)) + eps)
            reg = torch.sum(feature_norms)
            if self.scale_weights is not None:
                reg = reg + torch.sum(torch.sqrt(torch.sum(self.scale_weights**2, dim=-1) + eps))
            return reg

        elif self.penalty_type == "target_jacobian":
            if S_all is None:
                raise ValueError("S_all required for target_jacobian penalty")
            S_k = S_all[:, self.target_index]
            return torch.sqrt((S_k**2).mean() + eps)

        else:  # 'functional'
            if S_all is None:
                raise ValueError("S_all required for functional penalty")
            col_norms = torch.sqrt(torch.mean(S_all**2, dim=0) + eps)
            reg = torch.sum(col_norms)
            if self.scale_weights is not None:
                reg = reg + 0.05 * torch.sum(torch.sqrt(torch.sum(self.scale_weights**2, dim=-1) + eps))
            return reg

    def compute_loss(self, data, reg_value=0.0):
        """
        Computes regularized loss analytically in closed form:
            Loss = Mean( 0.5 * (S^k(x))^2 - log(partial_k S^k(x)) ) + reg_value * Penalty
        """
        derivs = self.compute_derivatives(data)
        out = derivs["out"]
        S_k = derivs["S_k"]
        S_all = derivs["S_all"]

        nll_loss = (0.5 * (out**2) - torch.log(S_k)).mean()
        reg_penalty = self.compute_regularization(S_all)
        total_loss = nll_loss + reg_value * reg_penalty

        return total_loss, nll_loss, reg_penalty

    def compute_precision_row(self, data):
        """
        Computes the k-th row of the generalized precision matrix Hat{Omega}_{j, k}
        analytically for all j = 0 ... d-1 using closed-form chain-rule derivatives.
        """
        self.eval()
        with torch.no_grad():
            derivs = self.compute_derivatives(data)
            out = derivs["out"]
            S_k = derivs["S_k"]
            S_all = derivs["S_all"]
            S_jk = derivs["S_jk"]
            S_kk = derivs["S_kk"]
            S_jkk = derivs["S_jkk"]

            term_score = -S_all * S_k - out * S_jk
            term_hess = (S_jkk * S_k - S_kk * S_jk) / (S_k**2)
            T_all = term_score + term_hess

            row = torch.mean(torch.abs(T_all), dim=0).cpu().numpy()
            row[self.target_index] = 1.0

        return row

    def compute_precision_row_split(self, data):
        self.eval()
        with torch.no_grad():
            derivs = self.compute_derivatives(data)
            out = derivs["out"]
            S_k = derivs["S_k"]
            S_all = derivs["S_all"]
            S_jk = derivs["S_jk"]
            S_kk = derivs["S_kk"]
            S_jkk = derivs["S_jkk"]

            term1 = -S_all * S_k - out * S_jk
            term2 = (S_jkk * S_k - S_kk * S_jk) / (S_k**2)

            row = torch.mean(torch.abs(term1) + torch.abs(term2), dim=0).cpu().numpy()
            row[self.target_index] = 1.0

        return row

    def compute_precision_row_torch_func(self, data):
        self.eval()
        x_data = data.to(self.device)

        def single_sample_loss(x_single):
            x_in = x_single.unsqueeze(0)
            derivs = self.compute_derivatives(x_in)
            out = derivs["out"].squeeze()
            S_k = derivs["S_k"].squeeze()
            return -0.5 * (out**2) + torch.log(S_k)

        hess_fn = torch.func.hessian(single_sample_loss)
        vmap_hess = torch.func.vmap(hess_fn)

        hessians = vmap_hess(x_data)
        row_vals = torch.mean(torch.abs(hessians[:, self.target_index, :]), dim=0).detach().cpu().numpy()
        row_vals[self.target_index] = 1.0
        return row_vals


class ScaleModulatedKANMap(MonotonicKANMap):
    """
    Subclass of MonotonicKANMap with conditioning_scale=True by default.
    Ideal for non-Gaussian distributions with multiplicative conditional scaling (e.g., Butterfly).
    """

    def __init__(self, *args, **kwargs):
        kwargs["conditioning_scale"] = True
        super().__init__(*args, **kwargs)
