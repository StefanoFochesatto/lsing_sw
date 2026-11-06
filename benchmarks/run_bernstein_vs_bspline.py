#!/usr/bin/env python3
"""
Comprehensive Comparative Benchmark: Bernstein Polynomials vs B-Splines for Monotonic KANs.

Evaluates:
1. Micro-benchmarks: Vectorized basis & derivative evaluation latency across batch sizes and degrees.
2. Full Model Derivatives: End-to-end Einstein summation recurrence latency (1st, 2nd, 3rd derivatives).
3. Numerical Precision: Max derivative errors against float64 autograd ground truth.
4. Transport Map Training: Optimization speed, NLL convergence, and precision matrix accuracy on 6D Mixed Coupling.
"""

import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from models.monotonic_kan import (
    BSplineBasis,
    BernsteinBasis,
    KANLayer,
    MonotonicKANMap,
    ScaleModulatedKANMap,
)
from trainers.kan_trainer import KANTrainer, KANPrecisionComputer
from datasets.mixed_coupling_generator import (
    generate_mixed_coupling_dataset,
    get_mixed_coupling_ground_truth,
)


def benchmark_basis_latency():
    """Profiles basis and derivative computation across batch sizes and orders."""
    print("\n" + "=" * 70)
    print("1. MICRO-BENCHMARK: BASIS & DERIVATIVE EVALUATION SPEED")
    print("=" * 70)

    batch_sizes = [256, 1024, 4096]
    degrees = [4, 8, 12]
    n_features = 10
    n_iters = 50

    results = {}

    for deg in degrees:
        results[f"degree_{deg}"] = {}
        grid_bs = BSplineBasis.build_grid(-4.0, 4.0, num_intervals=deg, degree=3)
        grid_bern = BernsteinBasis.build_grid(-4.0, 4.0, degree=deg)

        print(f"\n--- Degree / Grid Size: {deg} ---")
        print(f"{'Batch Size':>12} | {'B-Spline (ms)':>15} | {'Bernstein (ms)':>15} | {'Speedup':>10}")
        print("-" * 60)

        for B in batch_sizes:
            x = torch.randn(B, n_features)

            # Warmup
            for _ in range(5):
                _ = BSplineBasis.eval_basis(x, grid_bs, degree=3)
                _ = BernsteinBasis.eval_basis(x, grid_bern, degree=deg)

            # B-Spline
            t0 = time.perf_counter()
            for _ in range(n_iters):
                _ = BSplineBasis.eval_basis(x, grid_bs, degree=3)
            t_bs = (time.perf_counter() - t0) / n_iters * 1000.0

            # Bernstein
            t0 = time.perf_counter()
            for _ in range(n_iters):
                _ = BernsteinBasis.eval_basis(x, grid_bern, degree=deg)
            t_bern = (time.perf_counter() - t0) / n_iters * 1000.0

            speedup = t_bs / t_bern
            print(f"{B:>12} | {t_bs:>15.2f} | {t_bern:>15.2f} | {speedup:>9.2f}x")

            results[f"degree_{deg}"][f"batch_{B}"] = {
                "bspline_ms": round(t_bs, 3),
                "bernstein_ms": round(t_bern, 3),
                "speedup": round(speedup, 2),
            }

    return results


def benchmark_model_derivatives():
    """Profiles full compute_derivatives (1st, 2nd, 3rd derivatives) in MonotonicKANMap."""
    print("\n" + "=" * 70)
    print("2. FULL MODEL DERIVATIVE EVALUATION LATENCY (MonotonicKANMap, d=6, Q=4)")
    print("=" * 70)

    batch_sizes = [256, 1024, 2500]
    n_iters = 30
    d = 6
    Q = 4

    results = {}

    print(f"{'Batch Size':>12} | {'B-Spline Map (ms)':>18} | {'Bernstein Map (ms)':>18} | {'Speedup':>10}")
    print("-" * 65)

    for B in batch_sizes:
        x = torch.randn(B, d)

        model_bs = MonotonicKANMap(d, target_index=0, hidden_dim=Q, grid_size=8, basis_type="bspline")
        model_bern = MonotonicKANMap(d, target_index=0, hidden_dim=Q, grid_size=8, basis_type="bernstein", degree=8)

        # Warmup
        for _ in range(5):
            _ = model_bs.compute_derivatives(x)
            _ = model_bern.compute_derivatives(x)

        # B-spline
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = model_bs.compute_derivatives(x)
        t_bs = (time.perf_counter() - t0) / n_iters * 1000.0

        # Bernstein
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = model_bern.compute_derivatives(x)
        t_bern = (time.perf_counter() - t0) / n_iters * 1000.0

        speedup = t_bs / t_bern
        print(f"{B:>12} | {t_bs:>18.2f} | {t_bern:>18.2f} | {speedup:>9.2f}x")

        results[f"batch_{B}"] = {
            "bspline_ms": round(t_bs, 3),
            "bernstein_ms": round(t_bern, 3),
            "speedup": round(speedup, 2),
        }

    return results


def benchmark_numerical_precision():
    """Checks max absolute errors of closed-form derivatives against float64 autograd."""
    print("\n" + "=" * 70)
    print("3. NUMERICAL DERIVATIVE PRECISION VS FLOAT64 AUTOGRAD")
    print("=" * 70)

    d = 4
    torch.manual_seed(42)

    model_bs = MonotonicKANMap(d, target_index=1, hidden_dim=3, grid_size=8, basis_type="bspline").to(torch.float64)
    model_bern = MonotonicKANMap(d, target_index=1, hidden_dim=3, basis_type="bernstein", degree=8).to(torch.float64)

    model_bs.randomize_parameters_(std=0.3, delta_std=0.5)
    model_bern.randomize_parameters_(std=0.3, delta_std=0.5)

    x = torch.randn(10, d, dtype=torch.float64, requires_grad=True)

    def evaluate_max_errors(model):
        derivs = model.compute_derivatives(x)
        err_s_all, err_s_k, err_s_jk, err_s_kk, err_s_jkk = 0.0, 0.0, 0.0, 0.0, 0.0

        for b in range(x.shape[0]):
            y_b = derivs["out"][b, 0]
            g1 = torch.autograd.grad(y_b, x, retain_graph=True, create_graph=True)[0]
            err_s_all = max(err_s_all, (derivs["S_all"][b] - g1[b]).abs().max().item())
            err_s_k = max(err_s_k, (derivs["S_k"][b, 0] - g1[b, 1]).abs().max().item())

            g_k = g1[b, 1]
            g2 = torch.autograd.grad(g_k, x, retain_graph=True, create_graph=True)[0]
            err_s_jk = max(err_s_jk, (derivs["S_jk"][b] - g2[b]).abs().max().item())
            err_s_kk = max(err_s_kk, (derivs["S_kk"][b, 0] - g2[b, 1]).abs().max().item())

            g_kk = g2[b, 1]
            g3 = torch.autograd.grad(g_kk, x, retain_graph=True)[0]
            err_s_jkk = max(err_s_jkk, (derivs["S_jkk"][b] - g3[b]).abs().max().item())

        return {
            "max_err_S_all": err_s_all,
            "max_err_S_k": err_s_k,
            "max_err_S_jk": err_s_jk,
            "max_err_S_kk": err_s_kk,
            "max_err_S_jkk": err_s_jkk,
        }

    errs_bs = evaluate_max_errors(model_bs)
    errs_bern = evaluate_max_errors(model_bern)

    print(f"{'Quantity':>15} | {'B-Spline Max Error':>22} | {'Bernstein Max Error':>22}")
    print("-" * 65)
    for k in errs_bs:
        print(f"{k:>15} | {errs_bs[k]:>22.2e} | {errs_bern[k]:>22.2e}")

    return {"bspline": errs_bs, "bernstein": errs_bern}


def benchmark_transport_training(quick: bool = False, seed: int = 42):
    """Trains Monotonic KANs on 6D Mixed Coupling comparing B-splines vs Bernstein."""
    print("\n" + "=" * 70)
    print("4. END-TO-END TRAINING: 6D MIXED COUPLING BENCHMARK")
    print("=" * 70)

    torch.manual_seed(seed)
    np.random.seed(seed)

    if quick:
        n_trn, n_val, n_tst = 800, 300, 500
        max_epochs = 10
        print(f"Quick Mode: n_trn={n_trn}, n_val={n_val}, n_tst={n_tst}, epochs={max_epochs}")
    else:
        n_trn, n_val, n_tst = 2500, 1000, 1500
        max_epochs = 35
        print(f"Full Mode: n_trn={n_trn}, n_val={n_val}, n_tst={n_tst}, epochs={max_epochs}")

    full_data = generate_mixed_coupling_dataset(n_trn + n_val + n_tst, standardize=True, seed=seed)
    trn_data = full_data[:n_trn]
    val_data = full_data[n_trn : n_trn + n_val]
    tst_data = full_data[n_trn + n_val :]

    d = 6
    regs = [0.005]
    true_adj = get_mixed_coupling_ground_truth()

    results = {}

    for basis_name in ["bspline", "bernstein"]:
        print(f"\n--- Training ScaleModulatedKANMap ({basis_name.upper()}) ---")
        t0 = time.perf_counter()

        models = [
            ScaleModulatedKANMap(
                d,
                target_index=k,
                hidden_dim=4,
                grid_size=8,
                basis_type=basis_name,
                degree=8 if basis_name == "bernstein" else None,
            )
            for k in range(d)
        ]
        trainer = KANTrainer(models, d, hidden_dim=4, grid_size=8, max_epochs=max_epochs)

        trained_models = []
        nlls = {}
        for k in range(d):
            others = [j for j in range(d) if j != k]
            best_m, _, _, _ = trainer.train_model(trn_data, k, others, regs, val_data, verbose=False)
            trained_models.append(best_m)
            _, nll_k, _ = best_m.compute_loss(tst_data)
            nlls[f"node_{k}"] = float(nll_k.item())

        train_time = time.perf_counter() - t0
        mean_nll = float(np.mean(list(nlls.values())))

        comp = KANPrecisionComputer(trained_models, tst_data, d)
        raw_asym = comp.compute()
        norm_sym = comp.make_symmetric(raw_asym)

        # Off-diagonal error wrt true graph
        mask_off = ~np.eye(d, dtype=bool)
        frob_err = float(np.linalg.norm(norm_sym[mask_off] - true_adj[mask_off]))

        print(f"Total training time: {train_time:.2f} s")
        print(f"Mean test NLL:       {mean_nll:.4f}")
        print(f"Precision Frobenius: {frob_err:.4f}")

        results[basis_name] = {
            "train_time_sec": round(train_time, 2),
            "mean_test_nll": round(mean_nll, 4),
            "test_nll_per_node": {k: round(v, 4) for k, v in nlls.items()},
            "precision_frob_error": round(frob_err, 4),
            "precision_matrix": norm_sym.tolist(),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Bernstein vs B-Spline Comparative Benchmarks")
    parser.add_argument("--quick", action="store_true", help="Quick mode with fewer samples & epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="results/kan_benchmarks/bernstein_vs_bspline_results.json")
    args = parser.parse_args()

    out_path = REPO_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "basis_latency": benchmark_basis_latency(),
        "model_derivatives": benchmark_model_derivatives(),
        "numerical_precision": benchmark_numerical_precision(),
        "training_benchmark": benchmark_transport_training(quick=args.quick, seed=args.seed),
    }

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"RESULTS SAVED TO: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
