import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .basis import BSplineBasis
from .layer import KANLayer, DeepKAN


class MonotonicKANMap(nn.Module):
    """
    Modular Monotonic Kolmogorov-Arnold Network (KAN) Transport Map Component S^k(x):

        Standard Additive Mode (conditioning_scale=False):
            S^k(x) = S^k_{core}(x)

        Scale-Modulated Mode (conditioning_scale=True):
            S^k(x) = exp( s_{max} * tanh( sigma(x_{-k}) / s_{max} ) ) * S^k_{core}(x) + alpha(x_{-k})

    Guarantees strict monotonicity along target dimension x_k (partial_k S^k > 0) everywhere
    across arbitrary network depth L >= 2 and arbitrary layer widths [d, n_1, ..., n_{L-1}, 1]:
        partial_k S^k(x) = exp(sigma(x_{-k})) * partial_k S^k_{core}(x) > 0.

    Eliminates numerical quadrature via closed-form analytic forward evaluation, derivatives, and
    generalized precision matrix entries using programmatic Einstein summation recurrence.
    """

    def __init__(
        self,
        num_features: int,
        target_index: int = 0,
        hidden_dim: int = 4,
        layers_hidden: list[int] | tuple[int, ...] | None = None,
        scale_layers: list[int] | tuple[int, ...] | None = None,
        offset_layers: list[int] | tuple[int, ...] | None = None,
        grid_size: int = 8,
        grid_range: tuple[float, float] = (-4.0, 4.0),
        spline_order: int = 3,
        penalty_type: str = "functional",
        conditioning_scale: bool = False,
        scale_bound: float = 4.0,
        eps_mono: float = 1e-4,
        delta_mono: float | None = None,
        active_path: list[int] | tuple[int, ...] | None = None,
        dev: str | torch.device = "cpu",
    ):
        """
        Args:
            num_features: Total dimensionality d of the data.
            target_index: Index k of the target feature being transported.
            hidden_dim: Hidden dimension Q for default 2-layer topology [d, hidden_dim, 1].
            layers_hidden: Arbitrary layer dimensions for monotonic core S^k_{core}:
                           [d, n_1, n_2, ..., 1]. If None, defaults to [d, hidden_dim, 1].
            scale_layers: Layer dimensions for conditioning scale KAN sigma(x_{-k}):
                          [d-1, m_1, ..., 1]. If None, defaults to [d-1, 1] (single layer).
            offset_layers: Layer dimensions for conditioning offset KAN alpha(x_{-k}):
                           [d-1, p_1, ..., 1]. If None, defaults to [d-1, 1] (single layer).
            grid_size: Number of intervals in B-spline knot grid.
            grid_range: (grid_min, grid_max) for standardized domain.
            spline_order: Spline degree p (default 3 for cubic splines).
            penalty_type: Regularization penalty: 'functional', 'coeff_group', or 'target_jacobian'.
            conditioning_scale: Whether to enable scale modulation E(sigma) * S_{core} + alpha.
            scale_bound: Saturation bound s_{max} for smooth tanh scale modulation.
            eps_mono: Strict positive floor delta_mono added to softplus increments.
            delta_mono: Alias for eps_mono.
            active_path: Node sequence (i_0^*, i_1^*, ..., i_L^*) of the active backbone path P^*.
            dev: Target PyTorch device.
        """
        super(MonotonicKANMap, self).__init__()
        self.num_features = num_features
        self.target_index = target_index
        self.grid_size = grid_size
        self.grid_range = grid_range
        self.spline_order = spline_order
        self.penalty_type = penalty_type
        self.conditioning_scale = conditioning_scale
        self.scale_bound = float(scale_bound)
        self.eps_mono = float(delta_mono) if delta_mono is not None else float(eps_mono)
        self.device = torch.device(dev)
        self.other_indices = [j for j in range(num_features) if j != target_index]
        self.d_cond = len(self.other_indices)

        # Core layer dimensions: [d, n_1, ..., 1]
        if layers_hidden is None:
            self.layers_hidden = [num_features, hidden_dim, 1]
        else:
            self.layers_hidden = list(layers_hidden)
            if self.layers_hidden[0] != num_features or self.layers_hidden[-1] != 1:
                raise ValueError(
                    f"layers_hidden must start with num_features ({num_features}) and end with 1. Got: {self.layers_hidden}"
                )
        self.hidden_dim = self.layers_hidden[1]
        self.L = len(self.layers_hidden) - 1

        # Certified Active Backbone Path P^* = (i_0^*, i_1^*, ..., i_L^*)
        if active_path is None:
            self.active_path = (self.target_index,) + (0,) * self.L
        else:
            self.active_path = tuple(active_path)
            if (
                len(self.active_path) != self.L + 1
                or self.active_path[0] != self.target_index
                or self.active_path[-1] != 0
            ):
                raise ValueError(
                    f"active_path must have length {self.L + 1}, start with target_index ({self.target_index}), and end with 0. Got: {self.active_path}"
                )

        # Register knot grid and Greville abscissae buffers
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

        # Conditioning projection matrix for out-of-place vmap scatter
        if self.d_cond > 0:
            cond_proj = torch.zeros(self.d_cond, num_features, device=self.device)
            for idx, j in enumerate(self.other_indices):
                cond_proj[idx, j] = 1.0
            self.register_buffer("cond_proj", cond_proj)
        else:
            self.cond_proj = None

        # -------------------------------------------------------------
        # Monotonic Core KAN Layers: [d -> n_1 -> n_2 -> ... -> 1]
        # Layer 0: only target feature x_k is monotonic; conditioning are unconstrained.
        # Layers 1..L-1: all downstream edges are monotonic.
        # -------------------------------------------------------------
        self.core_layers = nn.ModuleList()
        for l in range(self.L):
            n_in = self.layers_hidden[l]
            n_out = self.layers_hidden[l + 1]
            if l == 0:
                mask = torch.zeros(n_out, n_in, dtype=torch.bool, device=self.device)
                mask[:, self.target_index] = True
            else:
                mask = torch.ones(n_out, n_in, dtype=torch.bool, device=self.device)

            self.core_layers.append(
                KANLayer(
                    in_dim=n_in,
                    out_dim=n_out,
                    grid_size=grid_size,
                    grid_range=grid_range,
                    spline_order=spline_order,
                    monotonic_mask=mask,
                    eps_mono=self.eps_mono,
                    bias=True,
                    dev=self.device,
                )
            )

        # -------------------------------------------------------------
        # Conditioning Scale & Offset Networks (DeepKAN)
        # -------------------------------------------------------------
        self.scale_layers = scale_layers
        self.offset_layers = offset_layers

        if self.conditioning_scale and self.d_cond > 0:
            scale_dims = list(scale_layers) if scale_layers is not None else [self.d_cond, 1]
            self.scale_net = DeepKAN(
                layers_hidden=scale_dims,
                grid_size=grid_size,
                grid_range=grid_range,
                spline_order=spline_order,
                dev=self.device,
            )

            offset_dims = list(offset_layers) if offset_layers is not None else [self.d_cond, 1]
            self.offset_net = DeepKAN(
                layers_hidden=offset_dims,
                grid_size=grid_size,
                grid_range=grid_range,
                spline_order=spline_order,
                dev=self.device,
            )
        else:
            self.scale_net = None
            self.offset_net = None

        # Initialize to Identity Map: S^k(x) approx x_k, partial_k S^k approx 1.0
        self._initialize_identity()

    def _initialize_identity(self):
        """
        Initializes control coefficients via Marsden's identity along active backbone path:
            S^k(x) approx x_k
            partial_k S^k(x) approx 1.0
            partial_j S^k(x) approx 0.0 for all j != k.
        """
        with torch.no_grad():
            n1 = self.layers_hidden[1]
            step0 = self.step_h / float(n1)
            init_delta0 = torch.log(
                torch.expm1(torch.tensor(max(step0 - self.eps_mono, 1e-5)))
            ).item()
            init_delta_h = torch.log(
                torch.expm1(torch.tensor(max(self.step_h - self.eps_mono, 1e-5)))
            ).item()

            # Layer 0: target feature distributed across n_1 hidden nodes
            l0 = self.core_layers[0]
            if l0.has_monotonic:
                l0.delta.fill_(-8.0)
                l0.c0.fill_(0.0)
                l0.delta[:, self.target_index].fill_(init_delta0)
                l0.c0[:, self.target_index].fill_((self.greville[0] / float(n1)).item())
            if l0.has_unconstrained:
                l0.unconstrained_weights.zero_()
            if l0.bias is not None:
                l0.bias.zero_()

            # Intermediate hidden layers (if any, for depth L > 2)
            for l in range(1, self.L - 1):
                in_node = self.active_path[l]
                out_node = self.active_path[l + 1]
                layer = self.core_layers[l]
                if layer.has_monotonic:
                    layer.delta.fill_(-8.0)
                    layer.c0.fill_(0.0)
                    layer.delta[out_node, in_node].fill_(init_delta_h)
                    layer.c0[out_node, in_node].fill_(self.greville[0].item())
                if layer.has_unconstrained:
                    layer.unconstrained_weights.zero_()
                if layer.bias is not None:
                    layer.bias.zero_()

            # Output layer (L - 1): sums incoming activations
            l_last = self.core_layers[-1]
            if l_last.has_monotonic:
                l_last.delta.fill_(init_delta_h)
                l_last.c0.fill_(self.greville[0].item())
            if l_last.has_unconstrained:
                l_last.unconstrained_weights.zero_()
            if l_last.bias is not None:
                l_last.bias.zero_()

            if self.scale_net is not None:
                self.scale_net.initialize_zeros()

            if self.offset_net is not None:
                self.offset_net.initialize_zeros()


    @property
    def scale_weights(self) -> torch.Tensor | None:
        if self.scale_net is not None and len(self.scale_net.layers) > 0:
            return self.scale_net.layers[0].unconstrained_weights.squeeze(0)
        return None

    @property
    def scale_bias(self) -> torch.Tensor | None:
        if self.scale_net is not None and len(self.scale_net.layers) > 0:
            return self.scale_net.layers[0].bias
        return None

    @property
    def offset_weights(self) -> torch.Tensor | None:
        if self.offset_net is not None and len(self.offset_net.layers) > 0:
            return self.offset_net.layers[0].unconstrained_weights.squeeze(0)
        return None

    @property
    def offset_bias(self) -> torch.Tensor | None:
        if self.offset_net is not None and len(self.offset_net.layers) > 0:
            return self.offset_net.layers[0].bias
        return None


    def get_inner_target_coeffs(self) -> torch.Tensor:
        """
        Returns control coefficients for the target variable x_k at layer 0.
        Shape: (n_1, n_bases).
        """
        return self.core_layers[0].get_coeffs()[:, self.target_index]

    def get_outer_coeffs(self) -> torch.Tensor:
        """
        Returns control coefficients for the final layer.
        Shape: (n_{L-1}, n_bases).
        """
        return self.core_layers[-1].get_coeffs().squeeze(0)

    def randomize_parameters_(self, std: float = 0.5, delta_std: float = 0.8):
        """
        Randomizes core, scale, and offset parameters away from identity initialization.
        """
        with torch.no_grad():
            for layer in self.core_layers:
                if layer.has_monotonic:
                    layer.delta.normal_(0, delta_std)
                if layer.has_unconstrained:
                    layer.unconstrained_weights.normal_(0, std)
            if self.scale_net is not None:
                for layer in self.scale_net.layers:
                    layer.unconstrained_weights.normal_(0, std)
            if self.offset_net is not None:
                for layer in self.offset_net.layers:
                    layer.unconstrained_weights.normal_(0, std)

    def _assemble_input(self, target_data: torch.Tensor, non_target_data: torch.Tensor | None = None) -> torch.Tensor:
        """
        Assembles input data into a single unified tensor of shape (B, d).
        Supports both unified (B, d) input and (target, non_target) split input.
        """
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

    def forward(self, target_data: torch.Tensor, non_target_data: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward evaluation S^k(x). Supports both split inputs (target_data, non_target_data)
        and unified single-tensor input x of shape (B, d).
        """
        x = self._assemble_input(target_data, non_target_data)
        acts = x
        for layer in self.core_layers:
            acts = layer(acts)
        out_core = acts

        if self.conditioning_scale and self.d_cond > 0:
            x_cond = x[:, self.other_indices]
            sigma_raw = self.scale_net(x_cond)
            offset = self.offset_net(x_cond)

            sigma = self.scale_bound * torch.tanh(sigma_raw / self.scale_bound)
            E = torch.exp(sigma)
            out = E * out_core + offset
        else:
            out = out_core

        return out

    def compute_derivatives(
        self, target_data: torch.Tensor, non_target_data: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Closed-form analytic computation of forward pass and all necessary derivatives
        across arbitrary depth L >= 2 using programmatic Einstein summation recurrence:
            - out: S^k(x) of shape (B, 1)
            - S_k: partial_k S^k(x) > 0 of shape (B, 1)
            - S_k_backbone: active backbone path derivative product D_{P^*}(x) of shape (B, 1)
            - S_all: partial_j S^k(x) for all j of shape (B, d)
            - S_jk: partial_j partial_k S^k(x) for all j of shape (B, d)
            - S_kk: partial_k^2 S^k(x) of shape (B, 1)
            - S_jkk: partial_j partial_k^2 S^k(x) for all j of shape (B, d)
        """
        x = self._assemble_input(target_data, non_target_data)
        B_size, d = x.shape

        acts = x
        layer_derivs = []
        for layer in self.core_layers:
            acts, p1, p2, p3 = layer.eval_splines_and_derivatives(acts)
            layer_derivs.append((p1, p2, p3))
        out_core = acts

        # -------------------------------------------------------------
        # Programmatic Einstein Summation Recurrence:
        # Initial conditions at input layer 0:
        #     A^{(0)} = I_d (shape: B, d, d)
        #     B^{(0)} = 0   (shape: B, d, d)
        #     D^{(0)} = 0   (shape: B, d, d)
        # -------------------------------------------------------------
        A = torch.eye(d, device=self.device, dtype=x.dtype).unsqueeze(0).expand(B_size, d, d)
        B_tens = torch.zeros(B_size, d, d, device=self.device, dtype=x.dtype)
        D_tens = torch.zeros(B_size, d, d, device=self.device, dtype=x.dtype)

        k = self.target_index
        for p1, p2, p3 in layer_derivs:
            A_k = A[:, :, k]  # (B, n_{l-1})
            C_k = B_tens[:, :, k]  # (B, n_{l-1})

            # 1. First derivatives: A^{(l)} = einsum("bai,bij->baj", p1, A^{(l-1)})
            A_next = torch.einsum("bai,bij->baj", p1, A)

            # 2. Mixed 2nd derivatives wrt (x_j, x_k):
            term_b1 = torch.einsum("bai,bi,bij->baj", p2, A_k, A)
            term_b2 = torch.einsum("bai,bij->baj", p1, B_tens)
            B_next = term_b1 + term_b2

            # 3. Mixed 3rd derivatives wrt (x_j, x_k^2):
            term_d1 = torch.einsum("bai,bi,bij->baj", p3, A_k**2, A)
            term_d2 = torch.einsum(
                "bai,bij->baj", p2, C_k.unsqueeze(-1) * A + 2.0 * A_k.unsqueeze(-1) * B_tens
            )
            term_d3 = torch.einsum("bai,bij->baj", p1, D_tens)
            D_next = term_d1 + term_d2 + term_d3

            A = A_next
            B_tens = B_next
            D_tens = D_next

        S_all_core = A.squeeze(1)  # (B, d)
        S_k_core = S_all_core[:, [k]]  # (B, 1)
        S_jk_core = B_tens.squeeze(1)  # (B, d)
        S_kk_core = S_jk_core[:, [k]]  # (B, 1)
        S_jkk_core = D_tens.squeeze(1)  # (B, d)

        # Active backbone path derivative product D_{P^*}(x) = prod_{l=1}^L phi'_{l, i_l^*, i_{l-1}^*}
        backbone_core = torch.ones(B_size, device=self.device, dtype=x.dtype)
        for l_idx, (p1_layer, _, _) in enumerate(layer_derivs):
            in_node = self.active_path[l_idx]
            out_node = self.active_path[l_idx + 1]
            backbone_core = backbone_core * p1_layer[:, out_node, in_node]

        # -------------------------------------------------------------
        # Scale Modulation Integration (if enabled)
        # -------------------------------------------------------------
        if self.conditioning_scale and self.d_cond > 0:
            x_cond = x[:, self.other_indices]
            sigma_raw, grad_sig_raw = self.scale_net.compute_forward_and_jacobian(x_cond)
            alpha, grad_alpha = self.offset_net.compute_forward_and_jacobian(x_cond)

            # Smooth tanh saturation: sigma = s_max * tanh(sigma_raw / s_max)
            tanh_val = torch.tanh(sigma_raw / self.scale_bound)
            sigma = self.scale_bound * tanh_val
            E = torch.exp(sigma)  # (B, 1)

            # d(sigma)/dx_j = (1 - tanh^2(sigma_raw / scale_bound)) * d(sigma_raw)/dx_j
            sech2 = 1.0 - tanh_val**2  # (B, 1)
            sig_d1 = sech2 * grad_sig_raw  # (B, d_cond)

            sigma_all = torch.matmul(sig_d1, self.cond_proj.to(dtype=x.dtype))
            alpha_all = torch.matmul(grad_alpha, self.cond_proj.to(dtype=x.dtype))

            out = E * out_core + alpha
            S_k = E * S_k_core
            S_k_backbone = E * backbone_core.unsqueeze(1)
            S_all = E * (sigma_all * out_core + S_all_core) + alpha_all
            S_all[:, [k]] = S_k

            S_jk = E * (sigma_all * S_k_core + S_jk_core)
            S_kk = E * S_kk_core
            S_jkk = E * (sigma_all * S_kk_core + S_jkk_core)
        else:
            out = out_core
            S_k = S_k_core
            S_k_backbone = backbone_core.unsqueeze(1)
            S_all = S_all_core
            S_jk = S_jk_core
            S_kk = S_kk_core
            S_jkk = S_jkk_core

        return {
            "out": out,
            "S_k": S_k,
            "S_k_backbone": S_k_backbone,
            "S_all": S_all,
            "S_jk": S_jk,
            "S_kk": S_kk,
            "S_jkk": S_jkk,
        }

    def compute_backbone_derivative(
        self, target_data: torch.Tensor, non_target_data: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Evaluates the exact 1D derivative product along the certified active backbone path P^*:
            D_{P^*}(x) = prod_{l=1}^L phi'_{l, i_l^*, i_{l-1}^*}(x^{(l-1)}_{i_{l-1}^*}) > 0.
        """
        derivs = self.compute_derivatives(target_data, non_target_data)
        return derivs["S_k_backbone"]

    def verify_backbone_monotonicity(
        self, data: torch.Tensor | None = None
    ) -> dict[str, object]:
        """
        Verifies that target derivative strictly dominates the active backbone path
        derivative everywhere: partial_k S^k(x) >= D_{P^*}(x) >= epsilon > 0.
        """
        if data is None:
            data = torch.randn(100, self.num_features, device=self.device)
        self.eval()
        with torch.no_grad():
            derivs = self.compute_derivatives(data)
            S_k = derivs["S_k"].squeeze()
            S_k_bb = derivs["S_k_backbone"].squeeze()
            is_valid = bool(
                torch.all(S_k >= S_k_bb - 1e-6).item() and torch.all(S_k_bb > 0).item()
            )
            return {
                "is_strictly_monotonic": is_valid,
                "S_k_min": float(S_k.min().item()),
                "backbone_min": float(S_k_bb.min().item()),
                "active_backbone_path": self.active_path,
            }

    def compute_regularization(self, S_all: torch.Tensor | None = None) -> torch.Tensor:
        """
        Computes regularization penalty:
            1. 'functional' (L-SING baseline Eq 5): sum_{j} sqrt( (1/M) sum_i (partial_j S^k(x^i))^2 )
            2. 'coeff_group' (KAN native): sum_{j != k} sqrt( sum_q sum_m |c_{q, j, m}|^2 )
            3. 'target_jacobian': sqrt( (partial_k S^k)^2.mean() )
        """
        eps = 1e-8
        if self.penalty_type == "coeff_group":
            reg = torch.tensor(0.0, device=self.device)
            unconstrained = self.core_layers[0].unconstrained_weights
            if unconstrained is not None and len(self.other_indices) > 0:
                cond_weights = unconstrained[:, self.other_indices, :]
                norms = torch.sqrt(torch.sum(cond_weights**2, dim=(0, 2)) + eps)
                reg = reg + torch.sum(norms)
            if self.scale_net is not None:
                for p in self.scale_net.parameters():
                    reg = reg + 0.01 * torch.norm(p)
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
            if self.scale_net is not None:
                for p in self.scale_net.parameters():
                    reg = reg + 0.01 * torch.norm(p)
            return reg

    def compute_loss(
        self, data: torch.Tensor, reg_value: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def compute_precision_row(self, data: torch.Tensor) -> np.ndarray:
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

    def compute_precision_row_split(self, data: torch.Tensor) -> np.ndarray:
        """
        Computes precision row with score and Hessian terms split (|A| + |B|).
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

            term1 = -S_all * S_k - out * S_jk
            term2 = (S_jkk * S_k - S_kk * S_jk) / (S_k**2)

            row = torch.mean(torch.abs(term1) + torch.abs(term2), dim=0).cpu().numpy()
            row[self.target_index] = 1.0

        return row

    def compute_precision_row_torch_func(self, data: torch.Tensor) -> np.ndarray:
        """
        Verification helper: evaluates precision row via PyTorch torch.func (vmap + hessian).
        """
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
    Supports arbitrary depth and hidden sizes for core, scale, and offset KANs.
    """

    def __init__(self, *args, **kwargs):
        kwargs["conditioning_scale"] = True
        super().__init__(*args, **kwargs)
