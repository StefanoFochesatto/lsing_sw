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
