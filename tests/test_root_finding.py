import pytest
import torch
from models.kan_transport import MonotonicKANMap, ScaleModulatedKANMap
from models.root_finding import invert_coordinate_map, invert_triangular_map


def test_1d_coordinate_inversion_convergence():
    torch.manual_seed(42)
    m = ScaleModulatedKANMap(num_features=1, target_index=0, hidden_dim=4, grid_size=8)
    for p in m.parameters():
        p.data.add_(torch.randn_like(p) * 0.1)

    z = torch.randn(200, 1)
    x_sol, diag = invert_coordinate_map(m, z, x_cond=None, tol=1e-5)

    assert diag["num_iters"] <= 6, f"Expected convergence in <= 6 iters, got {diag['num_iters']}"
    assert diag["final_max_residual"] < 2e-5
    assert diag["converged"].all()

    # Verify forward evaluation on solution matches target z
    with torch.no_grad():
        z_rec = m(x_sol)
        assert torch.allclose(z_rec, z, atol=2e-5)


def test_conditional_coordinate_inversion():
    torch.manual_seed(42)
    k = 3
    m = ScaleModulatedKANMap(num_features=k + 1, target_index=k, hidden_dim=4, grid_size=8)
    for p in m.parameters():
        p.data.add_(torch.randn_like(p) * 0.15)

    B = 250
    x_cond = torch.randn(B, k)
    z_k = torch.randn(B, 1)

    x_k, diag = invert_coordinate_map(m, z_k, x_cond=x_cond, tol=1e-5)

    assert diag["num_iters"] <= 6, f"Expected convergence in <= 6 iters, got {diag['num_iters']}"
    assert diag["final_max_residual"] < 2e-5
    assert diag["converged"].all()

    with torch.no_grad():
        z_rec = m(x_k, x_cond)
        assert torch.allclose(z_rec, z_k, atol=2e-5)


def test_extreme_targets_and_out_of_grid_extrapolation():
    torch.manual_seed(42)
    m = ScaleModulatedKANMap(num_features=2, target_index=1, hidden_dim=4, grid_size=8)
    for p in m.parameters():
        p.data.add_(torch.randn_like(p) * 0.1)

    # Large target z far outside the [-4, 4] grid
    z_k = torch.tensor([[-25.0], [-15.0], [0.0], [15.0], [25.0]], dtype=torch.float32)
    x_cond = torch.randn(5, 1)

    x_k, diag = invert_coordinate_map(m, z_k, x_cond=x_cond, tol=1e-4)

    assert diag["converged"].all()
    with torch.no_grad():
        z_rec = m(x_k, x_cond)
        assert torch.allclose(z_rec, z_k, atol=1e-4)


def test_full_triangular_map_roundtrip_inversion():
    torch.manual_seed(42)
    d = 4
    maps = []
    for k in range(d):
        m_k = ScaleModulatedKANMap(num_features=k + 1, target_index=k, hidden_dim=4, grid_size=8)
        for p in m_k.parameters():
            p.data.add_(torch.randn_like(p) * 0.1)
        maps.append(m_k)

    B = 100
    x_orig = torch.randn(B, d)

    # Forward pass x -> z
    z = torch.empty_like(x_orig)
    with torch.no_grad():
        for k in range(d):
            xk = x_orig[:, [k]]
            x_cond = x_orig[:, :k] if k > 0 else None
            z[:, [k]] = maps[k](xk, x_cond)

    # Inverse pass z -> x_recon
    x_recon, diags = invert_triangular_map(maps, z, tol=1e-5)

    assert len(diags) == d
    for k in range(d):
        assert diags[k]["num_iters"] <= 6

    max_diff = (x_orig - x_recon).abs().max().item()
    assert max_diff < 5e-5, f"Expected roundtrip diff < 5e-5, got {max_diff}"


def test_double_precision_quadratic_convergence():
    torch.manual_seed(42)
    m = ScaleModulatedKANMap(num_features=2, target_index=1, hidden_dim=4, grid_size=8).double()
    for p in m.parameters():
        p.data.add_(torch.randn_like(p) * 0.2)

    z = (torch.randn(50, 1) * 3.0).double()
    x_cond = (torch.randn(50, 1) * 2.0).double()

    x_sol, diag = invert_coordinate_map(m, z, x_cond=x_cond, tol=1e-10)

    assert diag["num_iters"] <= 7
    assert diag["final_max_residual"] < 1e-9
    assert diag["converged"].all()
