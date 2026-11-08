import math
import torch


class BSplineBasis:
    """
    Evaluates uniform B-spline basis functions and their analytic derivatives
    up to degree 3, with continuous linear tail extrapolation beyond [grid_min, grid_max].

    Provides C^2 continuity across knots, supporting up to 3rd-order mixed derivatives
    required for exact score and Hessian computations in transport map estimation.
    """

    @staticmethod
    def build_grid(
        grid_min: float = -4.0,
        grid_max: float = 4.0,
        num_intervals: int = 8,
        degree: int = 3,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Builds a uniform knot grid padded with `degree` knots on the left and right.

        Total knots: num_intervals + 1 + 2 * degree.
        Number of basis functions: num_intervals + degree.

        Args:
            grid_min: Lower bound of the active interpolation range.
            grid_max: Upper bound of the active interpolation range.
            num_intervals: Number of uniform grid intervals.
            degree: Spline degree (typically 3 for cubic splines).
            device: PyTorch device on which to construct the grid.
            dtype: Tensor data type.

        Returns:
            grid: 1D Tensor of knot positions with shape (num_intervals + 2 * degree + 1,).
        """
        step = (grid_max - grid_min) / float(num_intervals)
        inner = torch.linspace(grid_min, grid_max, num_intervals + 1, device=device, dtype=dtype)
        left = inner[0] - step * torch.arange(degree, 0, -1, device=device, dtype=dtype)
        right = inner[-1] + step * torch.arange(1, degree + 1, device=device, dtype=dtype)
        return torch.cat([left, inner, right])

    @staticmethod
    def get_greville_abscissae(grid: torch.Tensor, degree: int = 3) -> torch.Tensor:
        """
        Computes Greville abscissae (knot averages) for degree-p B-spline basis functions.

        For basis function i (i = 0, ..., n_bases - 1):
            xi_i = sum(grid[i + 1 : i + degree + 1]) / degree.

        Marsden's identity guarantees that sum_i xi_i * B_i(x) = x for all x in [grid_min, grid_max],
        enabling exact identity initialization of KAN layers.

        Args:
            grid: Knot grid tensor of shape (num_knots,).
            degree: Spline degree.

        Returns:
            greville: 1D Tensor of Greville abscissae of shape (n_bases,).
        """
        n_bases = len(grid) - degree - 1
        greville = torch.zeros(n_bases, device=grid.device, dtype=grid.dtype)
        for i in range(n_bases):
            greville[i] = grid[i + 1 : i + degree + 1].sum() / float(degree)
        return greville

    @staticmethod
    def eval_basis(
        x: torch.Tensor,
        grid: torch.Tensor,
        degree: int = 3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Vectorized evaluation of B-spline basis functions and their 1st, 2nd, and 3rd derivatives.

        Linear tail extrapolation is applied smoothly outside [grid_min, grid_max]:
        the 0th derivative continues linearly, the 1st derivative is constant, and
        the 2nd and 3rd derivatives vanish.

        Args:
            x: Input tensor of shape (...,).
            grid: Knot grid tensor of shape (num_knots,).
            degree: Spline degree (must be 3 for cubic splines).

        Returns:
            b_val: Basis values of shape (..., n_bases).
            b_d1: 1st derivatives with respect to x of shape (..., n_bases).
            b_d2: 2nd derivatives with respect to x of shape (..., n_bases).
            b_d3: 3rd derivatives with respect to x of shape (..., n_bases).
        """
        t_min = grid[degree]
        t_max = grid[-degree - 1]
        eps_boundary = 1e-6

        def _eval_interior(u: torch.Tensor):
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


class BernsteinBasis:
    """
    Evaluates Bernstein polynomial basis functions and their analytic derivatives
    up to degree p >= 3, with continuous linear tail extrapolation beyond [grid_min, grid_max].

    Key theoretical & computational properties:
    1. Partition of unity: sum_{i=0}^p B_{i, p}(u) = 1.0 for all u in [0, 1].
    2. Strict Monotonicity Transfer:
       phi'(x) = p / (b - a) * sum_{i=0}^{p-1} (c_{i+1} - c_i) B_{i, p-1}(u).
       Since B_{i, p-1}(u) >= 0 and sum B_{i, p-1}(u) = 1,
       c_{i+1} - c_i >= delta_mono > 0 guarantees phi'(x) >= p / (b - a) * delta_mono > 0.
    3. No Knot Grid:
       Parameterization depends only on the interval domain [grid_min, grid_max] and degree p.
       No internal knot divisions, intervals, or Cox-de Boor piecewise recursion.
    4. Exact Marsden-equivalent Identity:
       Setting c_i = a + (b - a) * i / p yields phi(x) == x, phi'(x) == 1, phi''(x) == 0,
       and phi'''(x) == 0 to machine precision.
    5. Fully Vectorized Derivative Calculation:
       Derivative basis vectors are computed via constant banded transformation matrices
       M_1 (p x p+1), M_2 (p-1 x p+1), and M_3 (p-2 x p+1) acting on lower-degree
       Bernstein bases in closed form without autograd.
    6. C^1 Linear Tail Extrapolation:
       For x < a and x > b, evaluates continuous linear extension matching boundary slopes,
       with zero 2nd and 3rd derivatives in the tails.
    """

    _CACHE: dict = {}

    @classmethod
    def _get_precomputed(cls, degree: int, scale: float, dtype: torch.dtype, device: torch.device):
        key = (degree, float(scale), dtype, str(device))
        if key not in cls._CACHE:
            p = degree
            # D1 maps (..., p) -> (..., p+1)
            D1 = torch.zeros(p, p + 1, dtype=dtype, device=device)
            for j in range(p):
                D1[j, j] = -p / scale
                D1[j, j + 1] = p / scale

            # D2 maps (..., p-1) -> (..., p)
            D2 = torch.zeros(p - 1, p, dtype=dtype, device=device)
            for j in range(p - 1):
                D2[j, j] = -(p - 1) / scale
                D2[j, j + 1] = (p - 1) / scale

            # D3 maps (..., p-2) -> (..., p-1)
            D3 = torch.zeros(p - 2, p - 1, dtype=dtype, device=device)
            for j in range(p - 2):
                D3[j, j] = -(p - 2) / scale
                D3[j, j + 1] = (p - 2) / scale

            M1 = D1
            M2 = D2 @ D1
            M3 = D3 @ D2 @ D1

            combs = [
                torch.tensor([math.comb(p - k, i) for i in range(p - k + 1)], dtype=dtype, device=device)
                for k in range(4)
            ]
            arange_p = torch.arange(p + 1, dtype=dtype, device=device)
            cls._CACHE[key] = (M1, M2, M3, combs, arange_p)
        return cls._CACHE[key]

    @staticmethod
    def build_grid(
        grid_min: float = -4.0,
        grid_max: float = 4.0,
        num_intervals: int | None = None,
        degree: int = 8,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Builds control point abscissae for Bernstein polynomials.
        Unlike B-splines, there is no knot grid; this returns the exact node positions
        xi_i = a + (b - a) * i / p for i in 0..p.

        Args:
            grid_min: Lower bound of active domain interval [a, b].
            grid_max: Upper bound of active domain interval [a, b].
            num_intervals: If provided, used as degree (for interface compatibility).
            degree: Polynomial degree p (default 8).
            device: Target PyTorch device.
            dtype: Tensor dtype.
        """
        deg = degree if num_intervals is None else num_intervals
        if deg < 3:
            raise ValueError(f"degree must be at least 3 for up to 3rd-order derivatives. Got: {deg}")
        return torch.linspace(grid_min, grid_max, deg + 1, device=device, dtype=dtype)

    @staticmethod
    def get_greville_abscissae(
        grid: torch.Tensor,
        degree: int | None = None,
    ) -> torch.Tensor:
        """
        Returns control node abscissae xi_i. For Bernstein polynomials, these are
        identically the node positions xi_i = a + (b - a) * i / p.
        """
        return grid.clone()

    @classmethod
    def eval_basis(
        cls,
        x: torch.Tensor,
        grid: torch.Tensor | tuple[float, float],
        degree: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluates degree-p Bernstein polynomial basis functions and their first 3
        analytic derivatives at input points x, with C^1 continuous linear tail
        extrapolation outside [grid_min, grid_max].

        Args:
            x: Input tensor of shape (B, ...) or (B, in_dim).
            grid: Node tensor or (grid_min, grid_max) tuple defining the domain.
            degree: Spline degree p. If None, inferred from len(grid) - 1.

        Returns:
            b_val: Basis values of shape (*x.shape, degree + 1).
            b_d1: 1st derivatives wrt x of shape (*x.shape, degree + 1).
            b_d2: 2nd derivatives wrt x of shape (*x.shape, degree + 1).
            b_d3: 3rd derivatives wrt x of shape (*x.shape, degree + 1).
        """
        if isinstance(grid, (tuple, list)):
            a, b = float(grid[0]), float(grid[1])
            p = 8 if degree is None else degree
        else:
            a, b = float(grid[0].item()), float(grid[-1].item())
            p = (len(grid) - 1) if degree is None else degree

        if p < 3:
            raise ValueError(f"degree must be at least 3 for up to 3rd-order derivatives. Got: {p}")

        scale = b - a
        dtype = x.dtype
        device = x.device

        M1, M2, M3, combs, arange_p = cls._get_precomputed(p, scale, dtype, device)

        u = (x - a) / scale
        u_clamped = torch.clamp(u, 0.0, 1.0)
        u_in = u_clamped.unsqueeze(-1)
        om_in = (1.0 - u_clamped).unsqueeze(-1)

        u_pows = u_in ** arange_p
        om_pows = om_in ** arange_p

        bases = []
        for k in range(4):
            d = p - k
            bases.append(combs[k] * u_pows[..., : d + 1] * torch.flip(om_pows[..., : d + 1], dims=[-1]))

        b_p = bases[0]
        b_d1 = bases[1] @ M1
        b_d2 = bases[2] @ M2
        b_d3 = bases[3] @ M3

        # Extrapolation masks
        is_left = (x <= a).unsqueeze(-1)
        is_right = (x >= b).unsqueeze(-1)
        dx_left = (x - a).unsqueeze(-1)
        dx_right = (x - b).unsqueeze(-1)

        # Boundary anchor values at x = a and x = b
        b_min = torch.zeros_like(b_p)
        b_min[..., 0] = 1.0
        db_min = torch.zeros_like(b_d1)
        db_min[..., 0] = -p / scale
        db_min[..., 1] = p / scale

        b_max = torch.zeros_like(b_p)
        b_max[..., -1] = 1.0
        db_max = torch.zeros_like(b_d1)
        db_max[..., -2] = -p / scale
        db_max[..., -1] = p / scale

        b_val = torch.where(
            is_left,
            b_min + db_min * dx_left,
            torch.where(is_right, b_max + db_max * dx_right, b_p),
        )
        b_d1 = torch.where(is_left, db_min, torch.where(is_right, db_max, b_d1))
        b_d2 = torch.where(is_left | is_right, torch.zeros_like(b_d2), b_d2)
        b_d3 = torch.where(is_left | is_right, torch.zeros_like(b_d3), b_d3)

        return b_val, b_d1, b_d2, b_d3


class IntegratedRBFBasis:
    """
    Evaluates Integrated Gaussian Radial Basis Functions (RBF) and their analytic
    derivatives up to 3rd order.
    """

    @staticmethod
    def build_grid(
        grid_min: float = -4.0,
        grid_max: float = 4.0,
        num_intervals: int | None = None,
        degree: int = 8,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Builds RBF centers linearly spaced across [grid_min, grid_max].
        """
        K = degree if num_intervals is None else num_intervals
        if K < 2:
            raise ValueError(f"Integrated RBF requires at least 2 centers. Got: {K}")
        return torch.linspace(grid_min, grid_max, K, device=device, dtype=dtype)

    @staticmethod
    def get_greville_abscissae(
        grid: torch.Tensor,
        degree: int | None = None,
    ) -> torch.Tensor:
        """
        Returns control node abscissae. Index 0 is 0.0 (carrier linear term),
        and indices 1..K are the Gaussian centers.
        """
        zero = torch.zeros(1, device=grid.device, dtype=grid.dtype)
        return torch.cat([zero, grid])

    @staticmethod
    def eval_basis(
        x: torch.Tensor,
        grid: torch.Tensor | tuple[float, float],
        degree: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluates Integrated Gaussian RBF basis functions and their first 3 derivatives.
        The basis includes a linear term at index 0: psi_0(x) = x, psi'_0 = 1, psi''_0 = 0, psi'''_0 = 0.
        Indices 1..K are integrated Gaussian RBFs:
            psi_m(x) = sigma * sqrt(pi / 2) * erf((x - mu_m) / (sqrt(2) * sigma)).
        """
        if isinstance(grid, (tuple, list)):
            a, b = float(grid[0]), float(grid[1])
            K = 8 if degree is None else degree
            mus = torch.linspace(a, b, K, dtype=x.dtype, device=x.device)
        else:
            mus = grid.to(dtype=x.dtype, device=x.device)
            a, b = float(mus[0].item()), float(mus[-1].item())
            K = len(mus)

        sigma = float((b - a) / max(K - 1, 1))
        c_erf = sigma * math.sqrt(math.pi / 2.0)

        diff = x.unsqueeze(-1) - mus
        z = diff / (math.sqrt(2.0) * sigma)

        psi_rbf = c_erf * torch.erf(z)
        g_rbf = torch.exp(-(diff ** 2) / (2.0 * (sigma ** 2)))
        d2_rbf = -(diff / (sigma ** 2)) * g_rbf
        d3_rbf = ((diff ** 2 - sigma ** 2) / (sigma ** 4)) * g_rbf

        x_in = x.unsqueeze(-1)
        ones = torch.ones_like(x_in)
        zeros = torch.zeros_like(x_in)

        b_val = torch.cat([x_in, psi_rbf], dim=-1)
        b_d1 = torch.cat([ones, g_rbf], dim=-1)
        b_d2 = torch.cat([zeros, d2_rbf], dim=-1)
        b_d3 = torch.cat([zeros, d3_rbf], dim=-1)

        return b_val, b_d1, b_d2, b_d3


class SOSChebyshevBasis:
    """
    Evaluates Sum-of-Squares (SOS) Chebyshev polynomial basis functions and their
    closed-form integrals and derivatives up to 3rd order.
    """

    _CACHE: dict = {}

    @classmethod
    def _get_precomputed(cls, degree: int, domain: tuple[float, float], dtype: torch.dtype, device: torch.device):
        key = (degree, float(domain[0]), float(domain[1]), dtype, str(device))
        if key not in cls._CACHE:
            K = max(1, degree // 2)
            N = 2 * K
            a, b = domain
            scale = float((b - a) / 2.0)
            mid = float((a + b) / 2.0)

            # Contraction tensor S of shape (N+1, K+1, K+1)
            S = torch.zeros(N + 1, K + 1, K + 1, dtype=dtype, device=device)
            for j in range(K + 1):
                for k in range(K + 1):
                    n1 = j + k
                    n2 = abs(j - k)
                    S[n1, j, k] += 0.5
                    S[n2, j, k] += 0.5

            # Constant offsets for Psi_n so Psi_n(0) = 0
            consts = torch.zeros(N + 1, dtype=dtype, device=device)
            for n in range(2, N + 1):
                consts[n] = 0.5 * (math.cos((n + 1) * math.pi / 2.0) / (n + 1) - math.cos((n - 1) * math.pi / 2.0) / (n - 1))

            cls._CACHE[key] = (K, N, scale, mid, S, consts)
        return cls._CACHE[key]

    @staticmethod
    def build_grid(
        grid_min: float = -4.0,
        grid_max: float = 4.0,
        num_intervals: int | None = None,
        degree: int = 6,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Builds Chebyshev-Lobatto nodes mapped to [grid_min, grid_max].
        """
        deg = degree if num_intervals is None else num_intervals
        K = max(1, deg // 2)
        N = 2 * K
        mid = (grid_min + grid_max) / 2.0
        scale = (grid_max - grid_min) / 2.0
        i = torch.arange(N + 1, device=device, dtype=dtype)
        u_nodes = torch.cos(i * math.pi / float(N))
        return mid + scale * u_nodes

    @staticmethod
    def get_greville_abscissae(grid: torch.Tensor, degree: int | None = None) -> torch.Tensor:
        return grid.clone()

    @classmethod
    def eval_basis(
        cls,
        x: torch.Tensor,
        grid: torch.Tensor | tuple[float, float],
        degree: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(grid, (tuple, list)):
            a, b = float(grid[0]), float(grid[1])
            deg = 6 if degree is None else degree
        else:
            a, b = float(grid.min().item()), float(grid.max().item())
            deg = (len(grid) - 1) if degree is None else degree

        dtype = x.dtype
        device = x.device
        K, N, scale, mid, S, consts = cls._get_precomputed(deg, (a, b), dtype, device)

        u = (x - mid) / scale
        u_clamped = torch.clamp(u, -1.0, 1.0)

        T = [torch.ones_like(u_clamped), u_clamped]
        dT = [torch.zeros_like(u_clamped), torch.ones_like(u_clamped)]
        d2T = [torch.zeros_like(u_clamped), torch.zeros_like(u_clamped)]
        d3T = [torch.zeros_like(u_clamped), torch.zeros_like(u_clamped)]
        for n in range(1, N + 1):
            T.append(2 * u_clamped * T[n] - T[n - 1])
            dT.append(2 * T[n] + 2 * u_clamped * dT[n] - dT[n - 1])
            d2T.append(4 * dT[n] + 2 * u_clamped * d2T[n] - d2T[n - 1])
            d3T.append(6 * d2T[n] + 2 * u_clamped * d3T[n] - d3T[n - 1])

        Psi = [u_clamped, 0.25 * T[2] - 0.25]
        for n in range(2, N + 1):
            Psi.append(0.5 * (T[n + 1] / (n + 1) - T[n - 1] / (n - 1)) - consts[n])

        b_val_int = torch.stack(Psi[:N + 1], dim=-1) * scale
        b_d1_int = torch.stack(T[:N + 1], dim=-1)
        b_d2_int = torch.stack(dT[:N + 1], dim=-1) / scale
        b_d3_int = torch.stack(d2T[:N + 1], dim=-1) / (scale ** 2)

        # Boundary anchors for extrapolation
        is_left = (x <= a).unsqueeze(-1)
        is_right = (x >= b).unsqueeze(-1)
        dx_left = (x - a).unsqueeze(-1)
        dx_right = (x - b).unsqueeze(-1)

        u_l = torch.tensor(-1.0, dtype=dtype, device=device)
        u_r = torch.tensor(1.0, dtype=dtype, device=device)

        T_l = [torch.ones_like(u_l), u_l]
        T_r = [torch.ones_like(u_r), u_r]
        for n in range(1, N + 1):
            T_l.append(2 * u_l * T_l[n] - T_l[n - 1])
            T_r.append(2 * u_r * T_r[n] - T_r[n - 1])

        Psi_l = [u_l, 0.25 * T_l[2] - 0.25]
        Psi_r = [u_r, 0.25 * T_r[2] - 0.25]
        for n in range(2, N + 1):
            Psi_l.append(0.5 * (T_l[n + 1] / (n + 1) - T_l[n - 1] / (n - 1)) - consts[n])
            Psi_r.append(0.5 * (T_r[n + 1] / (n + 1) - T_r[n - 1] / (n - 1)) - consts[n])

        b_min = torch.stack(Psi_l[:N + 1]) * scale
        b_max = torch.stack(Psi_r[:N + 1]) * scale
        db_min = torch.stack(T_l[:N + 1])
        db_max = torch.stack(T_r[:N + 1])

        b_val = torch.where(
            is_left,
            b_min + db_min * dx_left,
            torch.where(is_right, b_max + db_max * dx_right, b_val_int),
        )
        b_d1 = torch.where(is_left, db_min, torch.where(is_right, db_max, b_d1_int))
        b_d2 = torch.where(is_left | is_right, torch.zeros_like(b_d2_int), b_d2_int)
        b_d3 = torch.where(is_left | is_right, torch.zeros_like(b_d3_int), b_d3_int)

        return b_val, b_d1, b_d2, b_d3
