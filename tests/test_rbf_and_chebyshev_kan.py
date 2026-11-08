import pytest
import math
import torch
import torch.nn.functional as F
import numpy as np

from models.monotonic_kan import (
    MonotonicKANMap,
    ScaleModulatedKANMap,
    DeepKAN,
    IntegratedRBFBasis,
    SOSChebyshevBasis,
)


def test_rbf_basis_derivatives_vs_autograd():
    """Verify analytic basis derivatives of IntegratedRBFBasis up to 3rd order vs autograd."""
    grid = IntegratedRBFBasis.build_grid(-4.0, 4.0, degree=8, dtype=torch.float64)
    x = torch.tensor([-7.0, -3.2, -1.0, 0.0, 1.7, 4.0, 6.8], dtype=torch.float64, requires_grad=True)

    b_val, b_d1, b_d2, b_d3 = IntegratedRBFBasis.eval_basis(x, grid, degree=8)
    coeffs = torch.randn(9, dtype=torch.float64)

    y = (b_val * coeffs).sum()
    g1 = torch.autograd.grad(y, x, create_graph=True)[0]
    g1_an = (b_d1 * coeffs).sum(dim=-1)
    assert torch.allclose(g1, g1_an, atol=1e-12), "1st derivative mismatch for Integrated RBF"

    g2 = torch.autograd.grad(g1.sum(), x, create_graph=True)[0]
    g2_an = (b_d2 * coeffs).sum(dim=-1)
    assert torch.allclose(g2, g2_an, atol=1e-12), "2nd derivative mismatch for Integrated RBF"

    g3 = torch.autograd.grad(g2.sum(), x)[0]
    g3_an = (b_d3 * coeffs).sum(dim=-1)
    assert torch.allclose(g3, g3_an, atol=1e-11), "3rd derivative mismatch for Integrated RBF"


def test_rbf_asymptotic_linear_behavior():
    """Verify that Integrated RBF naturally transitions to linear behavior in far tails."""
    grid = IntegratedRBFBasis.build_grid(-3.0, 3.0, degree=6, dtype=torch.float64)
    far_x = torch.tensor([-30.0, -20.0, 20.0, 30.0], dtype=torch.float64)

    b_val, b_d1, b_d2, b_d3 = IntegratedRBFBasis.eval_basis(far_x, grid, degree=6)

    # In far tails, Gaussian derivatives must vanish and linear slope must be 1.0 on column 0
    assert torch.allclose(b_d1[:, 0], torch.ones(4, dtype=torch.float64)), "Linear slope must be 1.0"
    assert torch.allclose(b_d1[:, 1:], torch.zeros_like(b_d1[:, 1:]), atol=1e-15), "Gaussian tails must be 0"
    assert torch.allclose(b_d2, torch.zeros_like(b_d2), atol=1e-15), "2nd derivative must vanish in far tails"
    assert torch.allclose(b_d3, torch.zeros_like(b_d3), atol=1e-15), "3rd derivative must vanish in far tails"


def test_rbf_strict_monotonicity():
    """Assert partial_k S^k(x) > 0 for MonotonicKANMap with RBF basis across wide domain."""
    torch.manual_seed(42)
    model = MonotonicKANMap(
        num_features=4,
        target_index=1,
        hidden_dim=4,
        basis_type="rbf",
        degree=6,
    )
    model.randomize_parameters_(std=0.5, delta_std=1.0)

    x_test = (torch.rand(500, 4) - 0.5) * 40.0
    derivs = model.compute_derivatives(x_test)
    S_k = derivs["S_k"].squeeze()

    assert torch.all(S_k > 0), "RBF target derivative must be strictly positive everywhere"
    assert torch.min(S_k).item() > 1e-6, "RBF target derivative must not vanish"

    eps = 0.05
    x_plus = x_test.clone()
    x_plus[:, 1] += eps
    out1 = model(x_test)
    out2 = model(x_plus)
    assert torch.all(out2 > out1), "Map must be strictly monotonically increasing along target axis"


def test_chebyshev_basis_derivatives_vs_autograd():
    """Verify analytic basis derivatives of SOSChebyshevBasis up to 3rd order vs autograd."""
    grid = SOSChebyshevBasis.build_grid(-4.0, 4.0, degree=6, dtype=torch.float64)
    x = torch.tensor([-7.0, -4.0, -1.8, 0.0, 2.3, 4.0, 7.2], dtype=torch.float64, requires_grad=True)

    b_val, b_d1, b_d2, b_d3 = SOSChebyshevBasis.eval_basis(x, grid, degree=6)
    coeffs = torch.randn(7, dtype=torch.float64)

    y = (b_val * coeffs).sum()
    g1 = torch.autograd.grad(y, x, create_graph=True)[0]
    g1_an = (b_d1 * coeffs).sum(dim=-1)
    assert torch.allclose(g1, g1_an, atol=1e-12), "1st derivative mismatch for Chebyshev"

    # Interior points for 2nd and 3rd derivatives
    x_int = torch.tensor([-1.8, 0.0, 2.3], dtype=torch.float64, requires_grad=True)
    b_val_i, b_d1_i, b_d2_i, b_d3_i = SOSChebyshevBasis.eval_basis(x_int, grid, degree=6)
    y_i = (b_val_i * coeffs).sum()
    g1_i = torch.autograd.grad(y_i, x_int, create_graph=True)[0]
    g2_i = torch.autograd.grad(g1_i.sum(), x_int, create_graph=True)[0]
    g2_an = (b_d2_i * coeffs).sum(dim=-1)
    assert torch.allclose(g2_i, g2_an, atol=1e-11), "2nd derivative mismatch for Chebyshev"

    g3_i = torch.autograd.grad(g2_i.sum(), x_int)[0]
    g3_an = (b_d3_i * coeffs).sum(dim=-1)
    assert torch.allclose(g3_i, g3_an, atol=1e-10), "3rd derivative mismatch for Chebyshev"


def test_chebyshev_strict_monotonicity():
    """Assert partial_k S^k(x) > 0 for MonotonicKANMap with Chebyshev basis across wide domain."""
    torch.manual_seed(42)
    model = MonotonicKANMap(
        num_features=4,
        target_index=1,
        hidden_dim=4,
        basis_type="chebyshev",
        degree=6,
    )
    model.randomize_parameters_(std=0.3, delta_std=0.5)

    x_test = (torch.rand(500, 4) - 0.5) * 30.0
    derivs = model.compute_derivatives(x_test)
    S_k = derivs["S_k"].squeeze()

    assert torch.all(S_k > 0), "Chebyshev target derivative must be strictly positive everywhere"
    assert torch.min(S_k).item() > 1e-6, "Chebyshev target derivative must not vanish"


def test_rbf_closed_form_derivatives_vs_autograd():
    """Verify MonotonicKANMap with RBF basis closed-form derivatives against float64 autograd."""
    torch.manual_seed(42)
    model = MonotonicKANMap(
        num_features=4,
        target_index=1,
        hidden_dim=3,
        basis_type="rbf",
        degree=6,
        dev="cpu",
    ).to(torch.float64)

    model.randomize_parameters_(std=0.3, delta_std=0.5)
    x = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
    derivs = model.compute_derivatives(x)

    for b in range(6):
        y_b = derivs["out"][b, 0]
        grad_x = torch.autograd.grad(y_b, x, retain_graph=True, create_graph=True)[0]
        assert torch.allclose(derivs["S_all"][b], grad_x[b], atol=1e-10)
        assert torch.allclose(derivs["S_k"][b, 0], grad_x[b, 1], atol=1e-10)

        grad_k = grad_x[b, 1]
        grad2_xk = torch.autograd.grad(grad_k, x, retain_graph=True, create_graph=True)[0]
        assert torch.allclose(derivs["S_jk"][b], grad2_xk[b], atol=1e-9)
        assert torch.allclose(derivs["S_kk"][b, 0], grad2_xk[b, 1], atol=1e-9)

        grad_kk = grad2_xk[b, 1]
        grad3_xkk = torch.autograd.grad(grad_kk, x, retain_graph=True)[0]
        assert torch.allclose(derivs["S_jkk"][b], grad3_xkk[b], atol=1e-8)


def test_chebyshev_closed_form_derivatives_vs_autograd():
    """Verify MonotonicKANMap with Chebyshev basis closed-form derivatives against float64 autograd."""
    torch.manual_seed(42)
    model = MonotonicKANMap(
        num_features=4,
        target_index=1,
        hidden_dim=3,
        basis_type="chebyshev",
        degree=6,
        dev="cpu",
    ).to(torch.float64)

    model.randomize_parameters_(std=0.2, delta_std=0.4)
    x = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
    derivs = model.compute_derivatives(x)

    for b in range(6):
        y_b = derivs["out"][b, 0]
        grad_x = torch.autograd.grad(y_b, x, retain_graph=True, create_graph=True)[0]
        assert torch.allclose(derivs["S_all"][b], grad_x[b], atol=1e-10)
        assert torch.allclose(derivs["S_k"][b, 0], grad_x[b, 1], atol=1e-10)

        grad_k = grad_x[b, 1]
        grad2_xk = torch.autograd.grad(grad_k, x, retain_graph=True, create_graph=True)[0]
        assert torch.allclose(derivs["S_jk"][b], grad2_xk[b], atol=1e-9)
        assert torch.allclose(derivs["S_kk"][b, 0], grad2_xk[b, 1], atol=1e-9)

        grad_kk = grad2_xk[b, 1]
        grad3_xkk = torch.autograd.grad(grad_kk, x, retain_graph=True)[0]
        assert torch.allclose(derivs["S_jkk"][b], grad3_xkk[b], atol=1e-8)


def test_identity_initialization_all_four_bases():
    """Verify identity initialization S^k(x) approx x_k across all 4 bases."""
    torch.manual_seed(42)
    x = torch.randn(30, 4)

    for basis in ["bspline", "bernstein", "rbf", "chebyshev"]:
        model = MonotonicKANMap(
            num_features=4,
            target_index=2,
            hidden_dim=3,
            basis_type=basis,
            degree=6,
        )
        derivs = model.compute_derivatives(x)
        assert torch.allclose(derivs["out"].squeeze(), x[:, 2], atol=1e-3), f"Init identity failed for {basis}"
        assert torch.allclose(derivs["S_k"].squeeze(), torch.ones(30), atol=1e-3), f"Init S_k failed for {basis}"
