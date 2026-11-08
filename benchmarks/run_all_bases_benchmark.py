#!/usr/bin/env python3
"""
Comprehensive 4-Way Comparative Benchmark:
B-Splines vs Bernstein Polynomials vs Integrated RBF vs SOS-Chebyshev for Monotonic KANs.

Evaluates:
1. Micro-benchmarks: Basis & derivative evaluation latency across batch sizes and degrees.
2. Full Model Derivatives: Closed-form Einstein recurrence latency (1st, 2nd, 3rd derivatives).
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
    IntegratedRBFBasis,
    SOSChebyshevBasis,
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
    print("\n" + "=" * 80)
    print("1. MICRO-BENCHMARK: BASIS & DERIVATIVE EVALUATION SPEED (4 BASES)")
    print("=" * 80)

    batch_sizes = [256, 1024, 4096]
    n_features = 10
    n_iters = 50
    deg = 8

    grid_bs = BSplineBasis.build_grid(-4.0, 4.0, num_intervals=deg, degree=3)
    grid_bern = BernsteinBasis.build_grid(-4.0, 4.0, degree=deg)
    grid_rbf = IntegratedRBFBasis.build_grid(-4.0, 4.0, degree=deg)
    grid_cheb = SOSChebyshevBasis.build_grid(-4.0, 4.0, degree=deg)

    results = {}

    print(f"\nDegree / Grid Size = {deg}, Dimensions = {n_features}")
    print(f"{'Batch':>8} | {'B-Spline':>12} | {'Bernstein':>12} | {'Int-RBF':>12} | {'Chebyshev':>12} | {'Best Speedup':>14}")
    print("-" * 80)

    for B in batch_sizes:
        x = torch.randn(B, n_features)

        # Warmup
        for _ in range(5):
            _ = BSplineBasis.eval_basis(x, grid_bs, degree=3)
            _ = BernsteinBasis.eval_basis(x, grid_bern, degree=deg)
            _ = IntegratedRBFBasis.eval_basis(x, grid_rbf, degree=deg)
            _ = SOSChebyshevBasis.eval_basis(x, grid_cheb, degree=deg)

        times = {}
        for name, fn in [
            ("bspline", lambda: BSplineBasis.eval_basis(x, grid_bs, degree=3)),
            ("bernstein", lambda: BernsteinBasis.eval_basis(x, grid_bern, degree=deg)),
            ("rbf", lambda: IntegratedRBFBasis.eval_basis(x, grid_rbf, degree=deg)),
            ("chebyshev", lambda: SOSChebyshevBasis.eval_basis(x, grid_cheb, degree=deg)),
        ]:
            t0 = time.perf_counter()
            for _ in range(n_iters):
                _ = fn()
            times[name] = (time.perf_counter() - t0) / n_iters * 1000.0

        t_bs = times["bspline"]
        best_name = min(["bernstein", "rbf", "chebyshev"], key=lambda k: times[k])
        best_speedup = t_bs / times[best_name]

        print(
            f"{B:>8} | {times['bspline']:>10.2f}ms | {times['bernstein']:>10.2f}ms | "
            f"{times['rbf']:>10.2f}ms | {times['chebyshev']:>10.2f}ms | "
            f"{best_speedup:>9.2f}x ({best_name})"
        )

        results[f"batch_{B}"] = {k: round(v, 3) for k, v in times.items()}
        results[f"batch_{B}"]["best_speedup"] = round(best_speedup, 2)
        results[f"batch_{B}"]["best_basis"] = best_name

    return results


def benchmark_model_derivatives():
    print("\n" + "=" * 80)
    print("2. FULL MODEL DERIVATIVE EVALUATION LATENCY (MonotonicKANMap, d=6, Q=4)")
    print("=" * 80)

    batch_sizes = [256, 1024, 2500]
    n_iters = 30
    d = 6
    Q = 4

    results = {}

    print(f"{'Batch':>8} | {'B-Spline':>12} | {'Bernstein':>12} | {'Int-RBF':>12} | {'Chebyshev':>12} | {'Best Speedup':>14}")
    print("-" * 80)

    models = {
        "bspline": MonotonicKANMap(d, target_index=0, hidden_dim=Q, grid_size=8, basis_type="bspline"),
        "bernstein": MonotonicKANMap(d, target_index=0, hidden_dim=Q, grid_size=8, basis_type="bernstein", degree=8),
        "rbf": MonotonicKANMap(d, target_index=0, hidden_dim=Q, grid_size=8, basis_type="rbf", degree=8),
        "chebyshev": MonotonicKANMap(d, target_index=0, hidden_dim=Q, grid_size=8, basis_type="chebyshev", degree=8),
    }

    for B in batch_sizes:
        x = torch.randn(B, d)

        # Warmup
        for m in models.values():
            for _ in range(3):
                _ = m.compute_derivatives(x)

        times = {}
        for name, m in models.items():
            t0 = time.perf_counter()
            for _ in range(n_iters):
                _ = m.compute_derivatives(x)
            times[name] = (time.perf_counter() - t0) / n_iters * 1000.0

        t_bs = times["bspline"]
        best_name = min(["bernstein", "rbf", "chebyshev"], key=lambda k: times[k])
        best_speedup = t_bs / times[best_name]

        print(
            f"{B:>8} | {times['bspline']:>10.2f}ms | {times['bernstein']:>10.2f}ms | "
            f"{times['rbf']:>10.2f}ms | {times['chebyshev']:>10.2f}ms | "
            f"{best_speedup:>9.2f}x ({best_name})"
        )

        results[f"batch_{B}"] = {k: round(v, 3) for k, v in times.items()}
        results[f"batch_{B}"]["best_speedup"] = round(best_speedup, 2)
        results[f"batch_{B}"]["best_basis"] = best_name

    return results


def benchmark_numerical_precision():
    print("\n" + "=" * 80)
    print("3. NUMERICAL DERIVATIVE PRECISION VS FLOAT64 AUTOGRAD (ALL 4 BASES)")
    print("=" * 80)

    d = 4
    torch.manual_seed(42)

    bases = ["bspline", "bernstein", "rbf", "chebyshev"]
    models = {
        name: MonotonicKANMap(d, target_index=1, hidden_dim=3, grid_size=8, basis_type=name, degree=8).to(torch.float64)
        for name in bases
    }
    for m in models.values():
        m.randomize_parameters_(std=0.2, delta_std=0.4)

    x = torch.randn(8, d, dtype=torch.float64, requires_grad=True)

    def eval_errors(model):
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
            "S_all": err_s_all,
            "S_k": err_s_k,
            "S_jk": err_s_jk,
            "S_kk": err_s_kk,
            "S_jkk": err_s_jkk,
        }

    precision_results = {name: eval_errors(models[name]) for name in bases}

    print(f"{'Quantity':>10} | {'B-Spline':>14} | {'Bernstein':>14} | {'Int-RBF':>14} | {'Chebyshev':>14}")
    print("-" * 75)
    for q in ["S_all", "S_k", "S_jk", "S_kk", "S_jkk"]:
        print(
            f"{q:>10} | {precision_results['bspline'][q]:>14.2e} | "
            f"{precision_results['bernstein'][q]:>14.2e} | "
            f"{precision_results['rbf'][q]:>14.2e} | "
            f"{precision_results['chebyshev'][q]:>14.2e}"
        )

    return precision_results


def benchmark_transport_training(quick: bool = False, seed: int = 42):
    print("\n" + "=" * 80)
    print("4. END-TO-END TRAINING BENCHMARK: 6D MIXED COUPLING (4 BASES)")
    print("=" * 80)

    torch.manual_seed(seed)
    np.random.seed(seed)

    if quick:
        n_trn, n_val, n_tst = 600, 300, 400
        max_epochs = 8
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

    training_results = {}

    for basis_name in ["bspline", "bernstein", "rbf", "chebyshev"]:
        print(f"\n--- Training ScaleModulatedKANMap ({basis_name.upper()}) ---")
        t0 = time.perf_counter()

        models = [
            ScaleModulatedKANMap(
                d,
                target_index=k,
                hidden_dim=4,
                grid_size=8,
                basis_type=basis_name,
                degree=8 if basis_name in ["bernstein", "rbf"] else (6 if basis_name == "chebyshev" else None),
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

        mask_off = ~np.eye(d, dtype=bool)
        frob_err = float(np.linalg.norm(norm_sym[mask_off] - true_adj[mask_off]))

        print(f"Total training time: {train_time:.2f} s")
        print(f"Mean test NLL:       {mean_nll:.4f}")
        print(f"Precision Frobenius: {frob_err:.4f}")
        print(f"Active edge (0,1):   {norm_sym[0, 1]:.4f}")

        training_results[basis_name] = {
            "train_time_sec": round(train_time, 2),
            "mean_test_nll": round(mean_nll, 4),
            "test_nll_per_node": {k: round(v, 4) for k, v in nlls.items()},
            "precision_frob_error": round(frob_err, 4),
            "edge_0_1_signal": round(float(norm_sym[0, 1]), 4),
            "precision_matrix": norm_sym.tolist(),
        }

    return training_results


def main():
    parser = argparse.ArgumentParser(description="4-Way Comparative Basis Benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick mode for fast verification")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="results/kan_benchmarks/all_bases_benchmark_results.json")
    args = parser.parse_args()

    out_path = REPO_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "basis_latency": benchmark_basis_latency(),
        "model_derivatives": benchmark_model_derivatives(),
        "numerical_precision": benchmark_numerical_precision(),
        "training_benchmark": benchmark_transport_training(quick=args.quick, seed=args.seed),
    }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"ALL BENCHMARK RESULTS SAVED TO: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
