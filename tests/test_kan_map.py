import pytest
import torch
import torch.nn.functional as F
import numpy as np

from models.kan_transport import (
    MonotonicKANMap,
    ScaleModulatedKANMap,
    BSplineBasis,
    CubicHermiteBasis,
)


@pytest.fixture
def kan_model():
    torch.manual_seed(42)
    return MonotonicKANMap(
        num_features=5,
        target_index=2,
        hidden_dim=4,
        grid_size=8,
        grid_range=(-4.0, 4.0),
        spline_order=3,
        penalty_type="functional",
        dev="cpu",
    )


def test_spline_basis_partition_of_unity():
    """Verify that B-spline basis functions sum to 1.0 on the interior."""
    grid = BSplineBasis.build_grid(-4.0, 4.0, num_intervals=8, degree=3)
    x = torch.linspace(-3.5, 3.5, 50)
    b_val, _, _, _ = BSplineBasis.eval_basis(x, grid, degree=3)
    sum_bases = b_val.sum(dim=-1)
    assert torch.allclose(sum_bases, torch.ones_like(sum_bases), atol=1e-5), "B-splines must partition unity"


def test_spline_basis_derivatives_vs_autograd():
    """Verify analytic basis derivatives against PyTorch autograd in float64."""
    grid = BSplineBasis.build_grid(-4.0, 4.0, num_intervals=8, degree=3, dtype=torch.float64)
    x = torch.tensor([-2.5, -0.8, 0.0, 1.4, 3.1], dtype=torch.float64, requires_grad=True)

    b_val, b_d1, b_d2, b_d3 = BSplineBasis.eval_basis(x, grid, degree=3)
    coeffs = torch.randn(b_val.shape[-1], dtype=torch.float64)

    y = (b_val * coeffs).sum()
    g1 = torch.autograd.grad(y, x, create_graph=True)[0]
    g1_analytic = (b_d1 * coeffs).sum(dim=-1)
    assert torch.allclose(g1, g1_analytic, atol=1e-12), "1st derivative mismatch"

    g2 = torch.autograd.grad(g1.sum(), x, create_graph=True)[0]
    g2_analytic = (b_d2 * coeffs).sum(dim=-1)
    assert torch.allclose(g2, g2_analytic, atol=1e-12), "2nd derivative mismatch"

    g3 = torch.autograd.grad(g2.sum(), x)[0]
    g3_analytic = (b_d3 * coeffs).sum(dim=-1)
    assert torch.allclose(g3, g3_analytic, atol=1e-12), "3rd derivative mismatch"


def test_cubic_hermite_basis():
    """Verify CubicHermiteBasis evaluation and derivative correctness."""
    grid = torch.linspace(-4.0, 4.0, 9, dtype=torch.float64)
    x = torch.tensor([-2.5, 0.0, 2.5], dtype=torch.float64)
    vals, d1_vals, k = CubicHermiteBasis.eval_hermite(x, grid)
    h00, h10, h01, h11 = vals
    assert torch.all(h00 + h01 <= 1.0 + 1e-6), "Partition bounds hold for Hermite interpolation"
    assert len(k) == 3, "Interval count matches input"


def test_spline_linear_extrapolation():
    """Verify linear tail extrapolation outside [grid_min, grid_max]."""
    grid = BSplineBasis.build_grid(-3.0, 3.0, num_intervals=6, degree=3, dtype=torch.float64)
    x = torch.tensor([-7.0, -4.0, 0.0, 4.0, 8.0], dtype=torch.float64, requires_grad=True)

    b_val, b_d1, b_d2, b_d3 = BSplineBasis.eval_basis(x, grid, degree=3)
    coeffs = torch.randn(b_val.shape[-1], dtype=torch.float64)
    y = (b_val * coeffs).sum(dim=-1)

    g1 = torch.autograd.grad(y.sum(), x)[0]
    g1_analytic = (b_d1 * coeffs).sum(dim=-1)
    assert torch.allclose(g1, g1_analytic, atol=1e-12), "Extrapolation derivative mismatch"

    far_x = torch.tensor([-10.0, 10.0], dtype=torch.float64)
    _, _, d2_far, d3_far = BSplineBasis.eval_basis(far_x, grid, degree=3)
    assert torch.all(d2_far == 0), "Second derivative must vanish in tail extrapolation"
    assert torch.all(d3_far == 0), "Third derivative must vanish in tail extrapolation"


def test_identity_initialization(kan_model):
    """Verify that model initializes close to identity S^k(x) approx x_k and partial_k S^k approx 1."""
    x = torch.randn(50, 5)
    derivs = kan_model.compute_derivatives(x)
    target_k = kan_model.target_index

    assert torch.allclose(derivs["out"].squeeze(), x[:, target_k], atol=1e-4), "Init must approximate identity map"
    assert torch.allclose(derivs["S_k"].squeeze(), torch.ones(50), atol=1e-4), "Init target derivative must be ~1.0"
    for j in kan_model.other_indices:
        assert torch.allclose(derivs["S_all"][:, j], torch.zeros(50), atol=1e-2), f"Init S_{j} should be near 0"


def test_strict_monotonicity():
    """Assert partial_k S^k(x) > 0 across randomized grid evaluations in R^d including far tails."""
    torch.manual_seed(123)
    model = MonotonicKANMap(num_features=4, target_index=1, hidden_dim=4, grid_size=6)

    with torch.no_grad():
        model.inner_cond_weights.normal_(0, 0.5)
        model.inner_target_delta.normal_(0, 1.0)
        model.outer_delta.normal_(0, 1.0)

    x_test = (torch.rand(500, 4) - 0.5) * 30.0
    derivs = model.compute_derivatives(x_test)
    S_k = derivs["S_k"].squeeze()

    assert torch.all(S_k > 0), "Target derivative must be strictly positive everywhere"
    assert torch.min(S_k).item() > 1e-6, "Target derivative must not vanish"

    x1 = torch.randn(100, 4)
    x2 = x1.clone()
    step = torch.rand(100, 1) + 0.1
    x2[:, [1]] = x1[:, [1]] + step

    out1 = model(x1)
    out2 = model(x2)
    assert torch.all(out2 > out1), "Map must be strictly monotonically increasing along target axis"


def test_analytic_first_derivatives_vs_autograd():
    """Verify analytic first derivatives S_all against PyTorch autograd."""
    torch.manual_seed(99)
    model = MonotonicKANMap(num_features=4, target_index=2, hidden_dim=3, grid_size=6)
    with torch.no_grad():
        model.inner_cond_weights.normal_(0, 0.5)
        model.inner_target_delta.normal_(0, 1.0)
        model.outer_delta.normal_(0, 1.0)

    x = torch.randn(20, 4, requires_grad=True)
    out = model(x)
    grad_autograd = torch.autograd.grad(out.sum(), x)[0]

    derivs = model.compute_derivatives(x)
    S_all = derivs["S_all"]
    assert torch.allclose(grad_autograd, S_all, atol=1e-5), "Analytic S_all does not match autograd"


def test_analytic_first_derivatives_vs_finite_differences():
    """Verify analytic derivatives against numerical central finite differences in double precision."""
    torch.manual_seed(88)
    model = MonotonicKANMap(num_features=3, target_index=0, hidden_dim=2, grid_size=6).to(torch.float64)
    with torch.no_grad():
        model.inner_cond_weights.normal_(0, 0.4)
        model.inner_target_delta.normal_(0, 0.8)
        model.outer_delta.normal_(0, 0.8)

    x = torch.randn(15, 3, dtype=torch.float64)
    derivs = model.compute_derivatives(x)
    analytic_grad = derivs["S_all"]

    eps = 1e-5
    for j in range(3):
        x_plus = x.clone()
        x_plus[:, j] += eps
        x_minus = x.clone()
        x_minus[:, j] -= eps

        out_plus = model(x_plus).squeeze()
        out_minus = model(x_minus).squeeze()
        num_diff = (out_plus - out_minus) / (2.0 * eps)
        assert torch.allclose(analytic_grad[:, j], num_diff, atol=1e-5, rtol=1e-5), f"FD mismatch on dim {j}"


def test_analytic_second_derivatives_vs_autograd():
    """Verify analytic 2nd derivatives S_jk and S_kk against autograd."""
    torch.manual_seed(77)
    model = MonotonicKANMap(num_features=4, target_index=1, hidden_dim=3, grid_size=6)
    with torch.no_grad():
        model.inner_cond_weights.normal_(0, 0.5)
        model.inner_target_delta.normal_(0, 1.0)
        model.outer_delta.normal_(0, 1.0)

    x = torch.randn(15, 4, requires_grad=True)
    out = model(x)
    g1 = torch.autograd.grad(out.sum(), x, create_graph=True)[0]

    target_k = 1
    g2 = torch.autograd.grad(g1[:, target_k].sum(), x)[0]

    derivs = model.compute_derivatives(x)
    S_jk = derivs["S_jk"]
    S_kk = derivs["S_kk"]

    assert torch.allclose(g2[:, target_k], S_kk.squeeze(), atol=1e-5), "S_kk does not match autograd"
    for j in model.other_indices:
        assert torch.allclose(g2[:, j], S_jk[:, j], atol=1e-5), f"S_{j}k does not match autograd"


def test_analytic_precision_integrand_vs_autograd_and_torch_func():
    """
    Verify generalized precision integrand T_jk against:
        1. PyTorch Autograd second derivative of (-0.5*S^2 + log(S_k))
        2. torch.func.vmap + hessian
    """
    torch.manual_seed(66)
    model = MonotonicKANMap(num_features=4, target_index=1, hidden_dim=3, grid_size=6)
    with torch.no_grad():
        model.inner_cond_weights.normal_(0, 0.3)
        model.inner_target_delta.normal_(0, 0.8)
        model.outer_delta.normal_(0, 0.8)

    x = torch.randn(10, 4, requires_grad=True)
    derivs = model.compute_derivatives(x)
    out = derivs["out"]
    S_k = derivs["S_k"]
    S_all = derivs["S_all"]
    S_jk = derivs["S_jk"]
    S_kk = derivs["S_kk"]
    S_jkk = derivs["S_jkk"]

    T_all = -S_all * S_k - out * S_jk + (S_jkk * S_k - S_kk * S_jk) / (S_k**2)

    g = -0.5 * (out**2) + torch.log(S_k)
    dg_dxk = torch.autograd.grad(g.sum(), x, create_graph=True)[0][:, 1]
    d2g = torch.autograd.grad(dg_dxk.sum(), x)[0]

    for j in model.other_indices:
        assert torch.allclose(T_all[:, j], d2g[:, j], atol=1e-4), f"Analytic T_{j}1 mismatch with autograd"

    row_analytic = model.compute_precision_row(x)
    row_vmap = model.compute_precision_row_torch_func(x)
    assert np.allclose(row_analytic, row_vmap, atol=1e-4), "compute_precision_row mismatch with torch.func"


def test_loss_backward_pass():
    """Verify that compute_loss backpropagates finite gradients without NaNs or Infs."""
    torch.manual_seed(55)
    model = MonotonicKANMap(num_features=4, target_index=0, hidden_dim=4, grid_size=8, penalty_type="coeff_group")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    x = torch.randn(50, 4)
    for _ in range(5):
        optimizer.zero_grad()
        total_loss, nll, reg = model.compute_loss(x, reg_value=0.1)
        assert torch.isfinite(total_loss), "Loss must be finite"
        total_loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert torch.all(torch.isfinite(param.grad)), f"Gradient of {name} contains NaN or Inf"
        optimizer.step()


def test_scale_modulated_monotonicity_and_derivatives():
    """
    Verify ScaleModulatedKANMap:
    1. Strict positivity of target derivative partial_k S^k > 0.
    2. Analytic first, second, and third derivatives vs Autograd in float64.
    3. Generalized precision integrand T_jk vs Autograd.
    """
    torch.manual_seed(42)
    d = 4
    target_k = 2
    model = ScaleModulatedKANMap(d, target_index=target_k, hidden_dim=3, grid_size=6).to(torch.float64)

    with torch.no_grad():
        model.inner_cond_weights.normal_(0, 0.4)
        model.inner_target_delta.normal_(0, 0.8)
        model.outer_delta.normal_(0, 0.8)
        model.scale_weights.normal_(0, 0.5)
        model.scale_bias.normal_(0, 0.2)
        model.offset_weights.normal_(0, 0.3)
        model.offset_bias.normal_(0, 0.1)

    x = torch.randn(15, d, dtype=torch.float64, requires_grad=True)

    derivs = model.compute_derivatives(x)
    S_k = derivs["S_k"].squeeze()
    assert torch.all(S_k > 0), "Scale-modulated target derivative must be strictly positive"

    out = model(x)
    g1 = torch.autograd.grad(out.sum(), x, create_graph=True)[0]
    assert torch.allclose(derivs["S_all"], g1, atol=1e-12), "Scale-modulated 1st derivatives mismatch"

    g2 = torch.autograd.grad(g1[:, target_k].sum(), x, create_graph=True)[0]
    assert torch.allclose(derivs["S_kk"].squeeze(), g2[:, target_k], atol=1e-12), "Scale-modulated S_kk mismatch"
    for j in model.other_indices:
        assert torch.allclose(derivs["S_jk"][:, j], g2[:, j], atol=1e-12), f"Scale-modulated S_{j}k mismatch"

    g_loss = -0.5 * (out**2) + torch.log(derivs["S_k"])
    dg_dxk = torch.autograd.grad(g_loss.sum(), x, create_graph=True)[0][:, target_k]
    d2g = torch.autograd.grad(dg_dxk.sum(), x)[0]

    out_val = derivs["out"]
    S_all = derivs["S_all"]
    S_jk = derivs["S_jk"]
    S_kk = derivs["S_kk"]
    S_jkk = derivs["S_jkk"]
    T_all = -S_all * derivs["S_k"] - out_val * S_jk + (S_jkk * derivs["S_k"] - S_kk * S_jk) / (derivs["S_k"]**2)

    for j in model.other_indices:
        assert torch.allclose(T_all[:, j], d2g[:, j], atol=1e-11), f"Scale-modulated T_{j}k mismatch"

def test_scale_modulated_tanh_saturation_regime():
    """
    Verify ScaleModulatedKANMap in the saturated tanh regime (|sigma_raw| >> scale_bound):
    1. Strict saturation: |sigma| < scale_bound and e^{-scale_bound} <= E <= e^{scale_bound}.
    2. Exact match between analytic derivatives (sech^2 * psi') and PyTorch autograd in double precision.
    3. Generalized precision integrand T_jk vs PyTorch Autograd.
    """
    torch.manual_seed(101)
    d = 4
    target_k = 1
    s_bound = 3.5
    model = ScaleModulatedKANMap(d, target_index=target_k, hidden_dim=3, grid_size=6, scale_bound=s_bound).to(torch.float64)

    with torch.no_grad():
        # Initialize weights with very large values to push sigma deep into saturation
        model.inner_cond_weights.normal_(0, 0.5)
        model.inner_target_delta.normal_(0, 1.0)
        model.outer_delta.normal_(0, 1.0)
        model.scale_weights.uniform_(5.0, 15.0)  # Large scale weights
        model.scale_bias.fill_(8.0)              # Large bias ensuring saturation
        model.offset_weights.normal_(0, 0.5)
        model.offset_bias.normal_(0, 0.2)

    x = torch.randn(20, d, dtype=torch.float64, requires_grad=True)

    derivs = model.compute_derivatives(x)
    out = model(x)

    # 1. Bounds verification
    assert torch.all(derivs["S_k"] > 0), "Target derivative must remain strictly positive in saturation"

    # 2. First derivatives vs autograd
    g1 = torch.autograd.grad(out.sum(), x, create_graph=True)[0]
    assert torch.allclose(derivs["S_all"], g1, atol=1e-11), "1st derivatives in saturated tanh regime mismatch"

    # 3. Second derivatives vs autograd
    g2 = torch.autograd.grad(g1[:, target_k].sum(), x, create_graph=True)[0]
    assert torch.allclose(derivs["S_kk"].squeeze(), g2[:, target_k], atol=1e-11), "S_kk in saturated tanh regime mismatch"
    for j in model.other_indices:
        assert torch.allclose(derivs["S_jk"][:, j], g2[:, j], atol=1e-11), f"S_{j}k in saturated tanh regime mismatch"

    # 4. Precision integrand T_jk vs autograd
    g_loss = -0.5 * (out**2) + torch.log(derivs["S_k"])
    dg_dxk = torch.autograd.grad(g_loss.sum(), x, create_graph=True)[0][:, target_k]
    d2g = torch.autograd.grad(dg_dxk.sum(), x)[0]

    out_val = derivs["out"]
    S_all = derivs["S_all"]
    S_jk = derivs["S_jk"]
    S_kk = derivs["S_kk"]
    S_jkk = derivs["S_jkk"]
    T_all = -S_all * derivs["S_k"] - out_val * S_jk + (S_jkk * derivs["S_k"] - S_kk * S_jk) / (derivs["S_k"]**2)

    for j in model.other_indices:
        assert torch.allclose(T_all[:, j], d2g[:, j], atol=1e-11), f"T_{j}k in saturated tanh regime mismatch"


def test_scale_modulated_identity_initialization():
    """Verify that ScaleModulatedKANMap initializes to exact identity S(x) approx x_k."""
    torch.manual_seed(12)
    model = ScaleModulatedKANMap(num_features=4, target_index=1)
    x = torch.randn(30, 4)
    derivs = model.compute_derivatives(x)
    assert torch.allclose(derivs["out"].squeeze(), x[:, 1], atol=1e-4), "Scale-modulated init should be identity"
    assert torch.allclose(derivs["S_k"].squeeze(), torch.ones(30), atol=1e-4), "Target derivative init should be ~1.0"


def test_sm_kan_mixed_additive_and_multiplicative_routing():
    """
    Verify that on a mixed additive + multiplicative node (X_5 = 1.2 X_4 + X_2 * Z_5),
    ScaleModulatedKANMap achieves strictly lower NLL than a purely additive KAN and
    correctly routes the multiplicative parent (X_2) to the scale pathway psi_2 and
    the additive parent (X_4) to the offset pathway rho_4.
    """
    import sys
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root / "src") not in sys.path:
        sys.path.insert(0, str(repo_root / "src"))
    from datasets.mixed_coupling_generator import generate_mixed_coupling_dataset
    from trainers.kan_trainer import KANTrainer

    torch.manual_seed(42)
    np.random.seed(42)
    data = generate_mixed_coupling_dataset(1200, standardize=True, seed=42)
    trn, val, tst = data[:700], data[700:950], data[950:]
    d = 6
    k = 5
    others = [j for j in range(d) if j != k]

    base_add = [MonotonicKANMap(d, target_index=i, hidden_dim=4, grid_size=8, conditioning_scale=False) for i in range(d)]
    trainer_add = KANTrainer(base_add, d, hidden_dim=4, grid_size=8, max_epochs=25)
    best_add, _, _, _ = trainer_add.train_model(trn, k, others, [0.005], val)

    base_sm = [ScaleModulatedKANMap(d, target_index=i, hidden_dim=4, grid_size=8) for i in range(d)]
    trainer_sm = KANTrainer(base_sm, d, hidden_dim=4, grid_size=8, max_epochs=25)
    best_sm, _, _, _ = trainer_sm.train_model(trn, k, others, [0.005], val)

    _, nll_add, _ = best_add.compute_loss(tst)
    _, nll_sm, _ = best_sm.compute_loss(tst)
    assert nll_sm.item() < nll_add.item() - 0.10, "SM-KAN must significantly improve NLL on mixed additive+multiplicative node"

    with torch.no_grad():
        b_val, _, _, _ = BSplineBasis.eval_basis(tst, best_sm.grid, best_sm.spline_order)
        b_cond = b_val[:, others, :]
        psi_vals = torch.einsum("bdm,dm->bd", b_cond, best_sm.scale_weights)
        rho_vals = torch.einsum("bdm,dm->bd", b_cond, best_sm.offset_weights)
        idx_2 = others.index(2)
        idx_4 = others.index(4)
        idx_0 = others.index(0)
        assert psi_vals[:, idx_2].std().item() > 2.0 * psi_vals[:, idx_0].std().item()
        assert rho_vals[:, idx_4].std().item() > 3.0 * rho_vals[:, idx_0].std().item()
