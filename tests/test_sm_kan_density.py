import math
import pytest
import torch
from models.sm_kan_density import SMKANDensityEstimator


def test_sm_kan_density_initialization_and_forward():
    d = 3
    model = SMKANDensityEstimator(num_features=d, hidden_dim=4, grid_size=8)
    assert len(model.maps) == d
    for k, m in enumerate(model.maps):
        assert m.num_features == k + 1
        assert m.target_index == k

    x = torch.randn(20, d)
    z, log_det_j = model.forward(x)

    assert z.shape == (20, d)
    assert log_det_j.shape == (20,)


def test_sm_kan_log_prob_matches_autograd_jacobian():
    torch.manual_seed(42)
    d = 3
    model = SMKANDensityEstimator(num_features=d, hidden_dim=4, grid_size=8)

    # Perturb weights so it's strictly non-identity
    for p in model.parameters():
        p.data.add_(torch.randn_like(p) * 0.05)

    x_single = torch.randn(d)

    # Autograd Jacobian
    def f(v):
        z_out, _ = model.forward(v.unsqueeze(0))
        return z_out.squeeze(0)

    jac = torch.autograd.functional.jacobian(f, x_single)
    # Check lower triangular structure
    assert torch.allclose(jac.triu(diagonal=1), torch.zeros_like(jac), atol=1e-5)

    log_det_autograd = torch.log(torch.det(jac)).item()
    z_single, ldj_analytic = model.forward(x_single.unsqueeze(0))

    assert math.isclose(log_det_autograd, ldj_analytic.item(), rel_tol=1e-4, abs_tol=1e-5)

    # Check full log prob
    lp_expected = -0.5 * (d * math.log(2.0 * math.pi) + (z_single**2).sum().item()) + log_det_autograd
    lp_actual = model.log_prob(x_single.unsqueeze(0)).item()
    assert math.isclose(lp_expected, lp_actual, rel_tol=1e-4, abs_tol=1e-5)


def test_sm_kan_compute_loss_and_gradient_flow():
    torch.manual_seed(42)
    d = 4
    model = SMKANDensityEstimator(num_features=d, hidden_dim=4, grid_size=8)

    x = torch.randn(32, d)
    total_loss, nll_loss, reg_penalty = model.compute_loss(x, reg_value=0.01)

    assert total_loss.requires_grad
    assert nll_loss.requires_grad
    assert reg_penalty.requires_grad

    # Check equality of -log_prob.mean() and nll_loss
    lp_mean = -model.log_prob(x).mean()
    assert torch.allclose(lp_mean, nll_loss, atol=1e-5)

    # Test backward pass
    total_loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Gradient is None for {name}"
            assert not torch.isnan(p.grad).any(), f"NaN in gradient for {name}"


def test_sm_kan_invert_and_sample_roundtrip():
    torch.manual_seed(42)
    d = 3
    model = SMKANDensityEstimator(num_features=d, hidden_dim=4, grid_size=8)
    for p in model.parameters():
        p.data.add_(torch.randn_like(p) * 0.1)

    x_orig = torch.randn(30, d)
    z, _ = model.forward(x_orig)

    # Invert z -> x_recon
    x_recon, diags = model.invert(z, tol=1e-5)
    assert len(diags) == d

    max_diff = (x_orig - x_recon).abs().max().item()
    assert max_diff < 5e-5, f"Roundtrip inversion difference too large: {max_diff}"

    # Test sampling
    samples = model.sample(n_samples=25, tol=1e-5)
    assert samples.shape == (25, d)
    assert not torch.isnan(samples).any()


def test_sm_kan_standardization():
    torch.manual_seed(42)
    d = 2
    model = SMKANDensityEstimator(num_features=d, hidden_dim=4, grid_size=8)

    mean = torch.tensor([5.0, -10.0])
    std = torch.tensor([2.0, 3.0])
    model.set_standardization(mean, std)

    x = torch.randn(20, d) * std + mean
    lp = model.log_prob(x)
    assert lp.shape == (20,)
    assert not torch.isnan(lp).any()

    # Invert and sample with standardization
    z, _ = model.forward(x)
    x_recon, _ = model.invert(z, tol=1e-5)
    assert torch.allclose(x, x_recon, atol=5e-5)

    samples = model.sample(n_samples=10, tol=1e-5)
    assert samples.shape == (10, d)


def test_multistep_sm_kan_flow():
    torch.manual_seed(42)
    d = 3
    num_steps = 3
    model = SMKANDensityEstimator(num_features=d, num_steps=num_steps, hidden_dim=6, grid_size=8)
    
    # Check parameter count scaling
    single_model = SMKANDensityEstimator(num_features=d, num_steps=1, hidden_dim=6, grid_size=8)
    assert model.count_parameters() == single_model.count_parameters() * num_steps

    # Test forward and log_prob
    x = torch.randn(25, d)
    z, ldj = model.forward(x)
    assert z.shape == (25, d)
    assert ldj.shape == (25,)

    lp = model.log_prob(x)
    assert lp.shape == (25,)
    assert not torch.isnan(lp).any()

    # Test loss & gradient backprop
    loss, _, _ = model.compute_loss(x)
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None

    # Test roundtrip inversion across 3 stacked steps
    x_rec, diags = model.invert(z, tol=1e-5)
    assert len(diags) == d * num_steps
    diff = (x - x_rec).abs().max().item()
    assert diff < 5e-5, f"Multi-step roundtrip error too large: {diff}"

    # Test sampling
    samples = model.sample(15, tol=1e-5)
    assert samples.shape == (15, d)
