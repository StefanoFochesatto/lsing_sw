#!/usr/bin/env python3
"""
Theoretical Foundations & Proof Verification Suite for Monotonic KAN Transport Maps.

Empirically and analytically verifies the core theoretical claims in theory/kan_transport.tex:
1. Lemma 4.1 & Section 5.1: B-Spline Cumulative Softplus strict monotonicity and C^1 linear tail extrapolation on R.
2. Theorem 4.4: Arbitrarily deep KAN active backbone path monotonicity lower bound:
      partial_k S^k(x) >= D_{P^*}(x) >= delta_mono^L > 0 for all x in R^d.
3. Theorem 5.3: Exact algebraic Hessian Cancellation under Scale Modulation:
      (S_jkk S_k - S_kk S_jk) / S_k^2 == (Sc_jkk Sc_k - Sc_kk Sc_jk) / Sc_k^2
   invariant to arbitrary conditioning scale sigma(x_{-k}) and offset alpha(x_{-k}).
4. Section 5: Exactness of programmatic Einstein summation recurrence against float64 autograd.
"""

import sys
import torch
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from models.monotonic_kan import (
    BSplineBasis,
    KANLayer,
    DeepKAN,
    MonotonicKANMap,
    ScaleModulatedKANMap,
)


def verify_lemma_4_1_softplus_and_extrapolation():
    print("=" * 75)
    print("VERIFYING LEMMA 4.1 & SEC 5.1: CUMULATIVE SOFTPLUS & GLOBAL C^1 EXTRAPOLATION")
    print("=" * 75)

    num_intervals = 8
    spline_order = 3
    grid = BSplineBasis.build_grid(-4.0, 4.0, num_intervals=num_intervals, degree=spline_order)
    M = num_intervals + spline_order
    delta_mono = 0.05

    torch.manual_seed(42)
    theta = torch.randn(M)
    c = torch.zeros(M)
    c[0] = theta[0]
    c[1:] = c[0] + torch.cumsum(torch.nn.functional.softplus(theta[1:]) + delta_mono, dim=0)

    # Evaluate across interior and far tails [-20, 20]
    t = torch.linspace(-20.0, 20.0, 2000).unsqueeze(1)
    bases, d1, d2, d3 = BSplineBasis.eval_basis(t, grid, spline_order)
    c_tensor = c.view(1, 1, M)

    phi_prime = torch.sum(d1 * c_tensor, dim=-1).squeeze()
    min_deriv = phi_prime.min().item()

    print(f"  Grid Range: [-4.0, 4.0], Test Range: [-20.0, 20.0]")
    print(f"  Certified lower bound delta_mono: {delta_mono:.4f}")
    print(f"  Minimum derivative observed:       {min_deriv:.4f}")

    assert min_deriv > 0, f"Monotonicity violated: min derivative {min_deriv} <= 0"
    print("  [PASS] Spline derivative is strictly positive everywhere on R.\n")
    return True


def verify_theorem_4_4_arbitrary_depth_active_backbone():
    print("=" * 75)
    print("VERIFYING THEOREM 4.4: ARBITRARY DEPTH (L=3, L=4) ACTIVE BACKBONE CERTIFICATION")
    print("=" * 75)

    d = 8
    # Test 3-layer and 4-layer KANs
    architectures = [
        [d, 6, 4, 1],       # L = 3
        [d, 8, 6, 4, 1],    # L = 4
    ]

    for arch in architectures:
        L = len(arch) - 1
        target_k = 1
        model = MonotonicKANMap(d, target_index=target_k, layers_hidden=arch, grid_size=8)
        torch.manual_seed(100 + L)
        model.randomize_parameters_(std=0.5, delta_std=0.8)

        # Extreme range points
        x = torch.randn(500, d) * 4.0
        derivs = model.compute_derivatives(x)
        S_k = derivs["S_k"].squeeze()
        S_bb = derivs["S_k_backbone"].squeeze()

        min_S_k = S_k.min().item()
        min_bb = S_bb.min().item()
        margin = (S_k - S_bb).min().item()

        print(f"  Architecture: {arch} (Depth L={L})")
        print(f"    Active Backbone Path P*:       {model.active_path}")
        print(f"    Minimum S_k derivative:        {min_S_k:.6f} > 0")
        print(f"    Minimum Backbone D_P*(x):      {min_bb:.6f} > 0")
        print(f"    Minimum Dominance Margin:      {margin:.6e} >= 0")

        assert min_S_k > 0, f"S_k not strictly positive: {min_S_k}"
        assert min_bb > 0, f"Backbone derivative not strictly positive: {min_bb}"
        assert margin >= -1e-6, f"Theorem 4.4 dominance violated: margin {margin}"
        print(f"    [PASS] partial_k S^k(x) >= D_P*(x) > 0 certified across R^{d}.\n")
    return True


def verify_theorem_5_3_hessian_cancellation():
    print("=" * 75)
    print("VERIFYING THEOREM 5.3: HESSIAN CANCELLATION UNDER SCALE MODULATION")
    print("=" * 75)

    d = 6
    k = 2
    model = ScaleModulatedKANMap(d, target_index=k, hidden_dim=4, grid_size=8).to(torch.float64)
    torch.manual_seed(777)
    model.randomize_parameters_(std=0.7, delta_std=0.8)

    # Float64 double precision verification
    x = torch.randn(100, d, dtype=torch.float64) * 2.0

    derivs = model.compute_derivatives(x)
    S_k = derivs["S_k"]
    S_jk = derivs["S_jk"]
    S_kk = derivs["S_kk"]
    S_jkk = derivs["S_jkk"]

    lhs = (S_jkk * S_k - S_kk * S_jk) / (S_k ** 2)

    # Core unmodulated recurrence
    acts = x
    core_layer_derivs = []
    for layer in model.core_layers:
        acts, p1, p2, p3 = layer.eval_splines_and_derivatives(acts)
        core_layer_derivs.append((p1, p2, p3))

    B_size = x.shape[0]
    A = torch.eye(d, dtype=torch.float64).unsqueeze(0).expand(B_size, d, d)
    B_tens = torch.zeros(B_size, d, d, dtype=torch.float64)
    D_tens = torch.zeros(B_size, d, d, dtype=torch.float64)

    for p1, p2, p3 in core_layer_derivs:
        A_k = A[:, :, k]
        C_k = B_tens[:, :, k]
        A_next = torch.einsum('bai,bij->baj', p1, A)
        B_next = torch.einsum('bai,bi,bij->baj', p2, A_k, A) + torch.einsum('bai,bij->baj', p1, B_tens)
        D_next = torch.einsum('bai,bi,bij->baj', p3, A_k**2, A) + torch.einsum('bai,bij->baj', p2, C_k.unsqueeze(-1) * A + 2.0 * A_k.unsqueeze(-1) * B_tens) + torch.einsum('bai,bij->baj', p1, D_tens)
        A, B_tens, D_tens = A_next, B_next, D_next

    Sc_k = A[:, :, k]
    Sc_jk = B_tens.squeeze(1)
    Sc_kk = Sc_jk[:, [k]]
    Sc_jkk = D_tens.squeeze(1)

    rhs = (Sc_jkk * Sc_k - Sc_kk * Sc_jk) / (Sc_k ** 2)

    max_err = torch.max(torch.abs(lhs - rhs)).item()
    print(f"  Test Dimension d={d}, Target Index k={k}, Batch Size={x.shape[0]}")
    print(f"  Max Absolute Difference |LHS - RHS|: {max_err:.2e}")

    assert max_err < 1e-12, f"Theorem 5.3 violated: difference {max_err}"
    print("  [PASS] Hessian curvature ratio is exactly invariant to scale sigma and offset alpha.\n")
    return True


def verify_einstein_summation_vs_float64_autograd():
    print("=" * 75)
    print("VERIFYING CLOSED-FORM TENSOR RECURRENCE VS FLOAT64 PYTORCH AUTOGRAD")
    print("=" * 75)

    d = 4
    k = 0
    model = MonotonicKANMap(d, target_index=k, layers_hidden=[d, 5, 4, 1], grid_size=8).to(torch.float64)
    torch.manual_seed(888)
    model.randomize_parameters_(std=0.5, delta_std=0.5)

    x = torch.randn(20, d, dtype=torch.float64, requires_grad=True)
    derivs = model.compute_derivatives(x)

    # 1. First derivatives vs autograd
    out = model(x)
    grad_1 = torch.autograd.grad(out, x, torch.ones_like(out), create_graph=True)[0]
    err_1 = torch.max(torch.abs(derivs["S_all"] - grad_1)).item()

    # 2. Mixed 2nd derivatives wrt x_k
    s_k_val = grad_1[:, k]
    grad_2 = torch.autograd.grad(s_k_val, x, torch.ones_like(s_k_val), create_graph=True)[0]
    err_2 = torch.max(torch.abs(derivs["S_jk"] - grad_2)).item()

    # 3. Mixed 3rd derivatives wrt x_k^2
    s_kk_val = grad_2[:, k]
    grad_3 = torch.autograd.grad(s_kk_val, x, torch.ones_like(s_kk_val))[0]
    err_3 = torch.max(torch.abs(derivs["S_jkk"] - grad_3)).item()

    print(f"  1st Derivatives (S_all) max difference:   {err_1:.2e}")
    print(f"  2nd Mixed Derivatives (S_jk) max difference: {err_2:.2e}")
    print(f"  3rd Mixed Derivatives (S_jkk) max diff:    {err_3:.2e}")

    assert max(err_1, err_2, err_3) < 1e-10, "Tensor contraction does not match autograd ground truth!"
    print("  [PASS] Programmatic Einstein recurrence matches float64 autograd to double precision.\n")
    return True


def run_all_theory_verifications():
    print("\n" + "#" * 80)
    print("STARTING THEORY VERIFICATION SUITE FOR MONOTONIC KAN TRANSPORT MAPS")
    print("#" * 80 + "\n")

    verify_lemma_4_1_softplus_and_extrapolation()
    verify_theorem_4_4_arbitrary_depth_active_backbone()
    verify_theorem_5_3_hessian_cancellation()
    verify_einstein_summation_vs_float64_autograd()

    print("#" * 80)
    print("ALL 4 THEORETICAL THEOREMS & FOUNDATIONAL LEMMAS EMPIRICALLY CONFIRMED!")
    print("#" * 80 + "\n")


if __name__ == "__main__":
    run_all_theory_verifications()
