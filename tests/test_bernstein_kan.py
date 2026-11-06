import pytest
import torch
import torch.nn.functional as F
import numpy as np

from models.monotonic_kan import (
    MonotonicKANMap,
    ScaleModulatedKANMap,
    DeepKAN,
    BernsteinBasis,
    BSplineBasis,
)


@pytest.fixture
def bernstein_kan_model():
    torch.manual_seed(42)
    return MonotonicKANMap(
        num_features=5,
        target_index=2,
        hidden_dim=4,
        grid_size=8,
        basis_type="bernstein",
        degree=8,
        dev="cpu",
    )


def test_bernstein_basis_partition_of_unity():
    """Verify that Bernstein polynomial basis functions sum to 1.0 everywhere on [a, b]."""
    for deg in [4, 6, 8, 12]:
        grid = BernsteinBasis.build_grid(-4.0, 4.0, degree=deg, dtype=torch.float64)
        x = torch.linspace(-3.9, 3.9, 100, dtype=torch.float64)
        b_val, _, _, _ = BernsteinBasis.eval_basis(x, grid, degree=deg)
        sum_bases = b_val.sum(dim=-1)
        assert torch.allclose(
            sum_bases, torch.ones_like(sum_bases), atol=1e-12
        ), f"Bernstein basis p={deg} must partition unity"


def test_bernstein_basis_derivatives_vs_autograd():
    """Verify analytic basis derivatives of Bernstein polynomials up to 3rd order vs autograd."""
    for deg in [4, 6, 8]:
        grid = BernsteinBasis.build_grid(-4.0, 4.0, degree=deg, dtype=torch.float64)
        x = torch.tensor([-2.8, -1.2, 0.0, 1.5, 3.4], dtype=torch.float64, requires_grad=True)

        b_val, b_d1, b_d2, b_d3 = BernsteinBasis.eval_basis(x, grid, degree=deg)
        coeffs = torch.randn(deg + 1, dtype=torch.float64)

        y = (b_val * coeffs).sum()
        g1 = torch.autograd.grad(y, x, create_graph=True)[0]
        g1_analytic = (b_d1 * coeffs).sum(dim=-1)
        assert torch.allclose(
            g1, g1_analytic, atol=1e-12
        ), f"1st derivative mismatch for degree {deg}"

        g2 = torch.autograd.grad(g1.sum(), x, create_graph=True)[0]
        g2_analytic = (b_d2 * coeffs).sum(dim=-1)
        assert torch.allclose(
            g2, g2_analytic, atol=1e-11
        ), f"2nd derivative mismatch for degree {deg}"

        g3 = torch.autograd.grad(g2.sum(), x)[0]
        g3_analytic = (b_d3 * coeffs).sum(dim=-1)
        assert torch.allclose(
            g3, g3_analytic, atol=1e-10
        ), f"3rd derivative mismatch for degree {deg}"


def test_bernstein_linear_tail_extrapolation():
    """Verify C^1 continuous linear tail extrapolation outside [grid_min, grid_max]."""
    grid = BernsteinBasis.build_grid(-3.0, 3.0, degree=6, dtype=torch.float64)
    x = torch.tensor([-7.0, -4.5, -3.0, 0.0, 3.0, 4.5, 8.0], dtype=torch.float64, requires_grad=True)

    b_val, b_d1, b_d2, b_d3 = BernsteinBasis.eval_basis(x, grid, degree=6)
    coeffs = torch.randn(7, dtype=torch.float64)
    y = (b_val * coeffs).sum(dim=-1)

    g1 = torch.autograd.grad(y.sum(), x)[0]
    g1_analytic = (b_d1 * coeffs).sum(dim=-1)
    assert torch.allclose(g1, g1_analytic, atol=1e-11), "Extrapolation derivative mismatch"

    far_x = torch.tensor([-15.0, -5.0, 5.0, 15.0], dtype=torch.float64)
    _, _, d2_far, d3_far = BernsteinBasis.eval_basis(far_x, grid, degree=6)
    assert torch.all(d2_far == 0), "Second derivative must vanish in tail extrapolation"
    assert torch.all(d3_far == 0), "Third derivative must vanish in tail extrapolation"


def test_bernstein_exact_identity():
    """Verify that Bernstein control points c_i = a + (b - a)*i/p yield exact identity to float64 precision."""
    deg = 8
    a, b = -4.0, 4.0
    grid = BernsteinBasis.build_grid(a, b, degree=deg, dtype=torch.float64)
    c_identity = grid.clone()

    x = torch.linspace(-3.8, 3.8, 100, dtype=torch.float64)
    b_val, b_d1, b_d2, b_d3 = BernsteinBasis.eval_basis(x, grid, degree=deg)

    fx = (b_val * c_identity).sum(dim=-1)
    f_prime = (b_d1 * c_identity).sum(dim=-1)
    f_pprime = (b_d2 * c_identity).sum(dim=-1)

    assert torch.allclose(fx, x, atol=1e-14), "Must represent identity function exactly"
    assert torch.allclose(f_prime, torch.ones_like(f_prime), atol=1e-13), "Derivative must be 1.0 exactly"
    assert torch.allclose(f_pprime, torch.zeros_like(f_pprime), atol=1e-13), "Second derivative must be 0.0"


def test_bernstein_kan_identity_initialization(bernstein_kan_model):
    """Verify that MonotonicKANMap with Bernstein basis initializes to identity map."""
    x = torch.randn(50, 5)
    derivs = bernstein_kan_model.compute_derivatives(x)
    k = bernstein_kan_model.target_index

    assert torch.allclose(derivs["out"].squeeze(), x[:, k], atol=1e-4), "Init must approximate identity map"
    assert torch.allclose(derivs["S_k"].squeeze(), torch.ones(50), atol=1e-4), "Init target derivative must be ~1.0"
    for j in bernstein_kan_model.other_indices:
        assert torch.allclose(derivs["S_all"][:, j], torch.zeros(50), atol=1e-2), f"Init S_{j} should be near 0"


def test_bernstein_strict_monotonicity_transfer():
    """Assert partial_k S^k(x) > 0 across randomized parameters across R^d."""
    torch.manual_seed(42)
    model = MonotonicKANMap(
        num_features=4,
        target_index=1,
        hidden_dim=4,
        basis_type="bernstein",
        degree=6,
    )
    model.randomize_parameters_(std=0.5, delta_std=1.0)

    # Wide range including far tails
    x_test = (torch.rand(500, 4) - 0.5) * 30.0
    derivs = model.compute_derivatives(x_test)
    S_k = derivs["S_k"].squeeze()

    assert torch.all(S_k > 0), "Target derivative must be strictly positive everywhere"
    assert torch.min(S_k).item() > 1e-6, "Target derivative must not vanish"

    # Perturb target feature
    eps = 0.05
    x_plus = x_test.clone()
    x_plus[:, 1] += eps
    out1 = model(x_test)
    out2 = model(x_plus)
    assert torch.all(out2 > out1), "Map must be strictly monotonically increasing along target axis"


def test_bernstein_kan_closed_form_derivatives_vs_autograd():
    """Verify MonotonicKANMap closed-form derivatives against float64 PyTorch autograd."""
    torch.manual_seed(42)
    model = MonotonicKANMap(
        num_features=4,
        target_index=1,
        hidden_dim=3,
        basis_type="bernstein",
        degree=5,
        dev="cpu",
    ).to(torch.float64)

    model.randomize_parameters_(std=0.3, delta_std=0.5)

    x = torch.randn(8, 4, dtype=torch.float64, requires_grad=True)
    derivs = model.compute_derivatives(x)

    # 1. Check out = S^k(x)
    out_forward = model(x)
    assert torch.allclose(derivs["out"], out_forward, atol=1e-12)

    # 2. Check S_all = d(S^k)/dx
    for b in range(8):
        y_b = derivs["out"][b, 0]
        grad_x = torch.autograd.grad(y_b, x, retain_graph=True, create_graph=True)[0]
        assert torch.allclose(derivs["S_all"][b], grad_x[b], atol=1e-10), f"S_all mismatch at sample {b}"
        assert torch.allclose(derivs["S_k"][b, 0], grad_x[b, 1], atol=1e-10), f"S_k mismatch at sample {b}"

        # 3. Check S_jk and S_kk
        grad_k = grad_x[b, 1]
        grad2_xk = torch.autograd.grad(grad_k, x, retain_graph=True, create_graph=True)[0]
        assert torch.allclose(derivs["S_jk"][b], grad2_xk[b], atol=1e-9), f"S_jk mismatch at sample {b}"
        assert torch.allclose(derivs["S_kk"][b, 0], grad2_xk[b, 1], atol=1e-9), f"S_kk mismatch at sample {b}"

        # 4. Check S_jkk
        grad_kk = grad2_xk[b, 1]
        grad3_xkk = torch.autograd.grad(grad_kk, x, retain_graph=True)[0]
        assert torch.allclose(derivs["S_jkk"][b], grad3_xkk[b], atol=1e-8), f"S_jkk mismatch at sample {b}"


def test_bernstein_scale_modulated_kan_derivatives_vs_autograd():
    """Verify ScaleModulatedKANMap closed-form derivatives against float64 PyTorch autograd."""
    torch.manual_seed(42)
    model = ScaleModulatedKANMap(
        num_features=4,
        target_index=2,
        hidden_dim=3,
        basis_type="bernstein",
        degree=5,
        dev="cpu",
    ).to(torch.float64)

    model.randomize_parameters_(std=0.2, delta_std=0.4)

    x = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
    derivs = model.compute_derivatives(x)

    for b in range(6):
        y_b = derivs["out"][b, 0]
        grad_x = torch.autograd.grad(y_b, x, retain_graph=True, create_graph=True)[0]
        assert torch.allclose(derivs["S_all"][b], grad_x[b], atol=1e-10)
        assert torch.allclose(derivs["S_k"][b, 0], grad_x[b, 2], atol=1e-10)

        grad_k = grad_x[b, 2]
        grad2_xk = torch.autograd.grad(grad_k, x, retain_graph=True, create_graph=True)[0]
        assert torch.allclose(derivs["S_jk"][b], grad2_xk[b], atol=1e-9)
        assert torch.allclose(derivs["S_kk"][b, 0], grad2_xk[b, 2], atol=1e-9)

        grad_kk = grad2_xk[b, 2]
        grad3_xkk = torch.autograd.grad(grad_kk, x, retain_graph=True)[0]
        assert torch.allclose(derivs["S_jkk"][b], grad3_xkk[b], atol=1e-8)


def test_bernstein_deep_kan_jacobian():
    """Verify DeepKAN input Jacobian with Bernstein basis against autograd."""
    torch.manual_seed(42)
    deep_kan = DeepKAN(
        layers_hidden=[3, 4, 3, 1],
        basis_type="bernstein",
        degree=6,
        dev="cpu",
    ).to(torch.float64)

    with torch.no_grad():
        for layer in deep_kan.layers:
            layer.unconstrained_weights.normal_(0, 0.2)

    x = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    out, J_analytic = deep_kan.compute_forward_and_jacobian(x)

    for b in range(5):
        y_b = out[b, 0]
        grad_x = torch.autograd.grad(y_b, x, retain_graph=True)[0]
        assert torch.allclose(J_analytic[b], grad_x[b], atol=1e-10), f"Jacobian mismatch at sample {b}"
