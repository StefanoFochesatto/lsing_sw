#!/usr/bin/env python3
"""
Comprehensive Benchmarking Suite: Monotonic KAN Transport Map vs UMNN for L-SING.

Experiments covered:
1. Gaussian Benchmark (d=10): Edge recovery (F1, FPR, Precision, Recall at tau in {0.2, 0.1, 0.05}),
   training wall-clock time, memory footprint, and precision evaluation speed.
2. Butterfly Benchmark (d=10): Non-Gaussian nonlinear dependency recovery, sample size sensitivity
   (M in {500, 1000, 5000}), training time per epoch, and edge recovery metrics.
3. Butterfly Benchmark (d=40): High-dimensional scalability test (d=40, 20 true edges).
4. Profiling & Quadrature Overhead: Forward evaluation throughput, precision matrix evaluation throughput
   (analytic vs torch.func vs UMNN quadrature), parameter count, and memory overhead.
5. Sparsity Penalty Ablation: Functional penalty (Eq 5) vs Coefficient Group Lasso (KAN native).
"""

import os
import sys
import time
import json
import tracemalloc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Add project root and src to path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from models.UMNN import MonotonicNN
from models.monotonic_kan import MonotonicKANMap, BSplineBasis
from trainers.map_trainer import MapTrainer
from trainers.kan_trainer import KANTrainer, KANPrecisionComputer
from evaluators.precision_matrix_computer import PrecisionMatrixComputer
from datasets.butterfly_generator import generate_butterfly
import argparse
from models.monotonic_kan import ScaleModulatedKANMap
from datasets.mixed_coupling_generator import (
    generate_mixed_coupling_dataset,
    get_mixed_coupling_ground_truth,
)
from verify_kan_theory import run_all_theory_verifications


def compute_graph_metrics(precision_matrix, true_adj, thresholds=(0.2, 0.1, 0.05)):
    """
    Computes graph recovery metrics (TP, FP, FN, TN, Precision, Recall, F1, FPR)
    across given threshold values tau.
    """
    d = precision_matrix.shape[0]
    triu_idx = np.triu_indices(d, k=1)
    true_edges = true_adj[triu_idx].astype(bool)
    n_pos = int(np.sum(true_edges))
    n_neg = int(len(true_edges) - n_pos)

    results = {}
    for tau in thresholds:
        est_edges = (np.abs(precision_matrix[triu_idx]) > tau)
        tp = int(np.sum(est_edges & true_edges))
        fp = int(np.sum(est_edges & ~true_edges))
        fn = int(np.sum(~est_edges & true_edges))
        tn = int(np.sum(~est_edges & ~true_edges))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        fpr = fp / n_neg if n_neg > 0 else 0.0

        results[tau] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "fpr": float(fpr),
        }
    return results


def get_gaussian_ground_truth(d=10):
    """
    Returns true binary adjacency matrix for the Gaussian dataset from the paper.
    Edges: (0, 4), (2, 6), (5, 7), (6, 9) in 0-indexed.
    """
    adj = np.zeros((d, d), dtype=int)
    edges = [(0, 4), (2, 6), (5, 7), (6, 9)]
    for u, v in edges:
        adj[u, v] = adj[v, u] = 1
    return adj


def get_butterfly_ground_truth(d=10):
    """
    Returns true binary adjacency matrix for Butterfly dataset with pairs (2i, 2i+1).
    """
    adj = np.zeros((d, d), dtype=int)
    num_pairs = d // 2
    for i in range(num_pairs):
        u, v = 2 * i, 2 * i + 1
        adj[u, v] = adj[v, u] = 1
    return adj


def train_umnn_models(
    training_data,
    validation_data,
    d,
    hidden_layers=(16, 32),
    num_steps=50,
    regularizations=(1.0, 0.1, 0.01, 0.001, 0.0),
    learning_rate=0.01,
    max_epochs=50,
    dev="cpu",
):
    """Trains UMNN models across all d features."""
    base_map = [MonotonicNN(d, list(hidden_layers), num_steps, dev) for _ in range(d)]
    trainer = MapTrainer(base_map, d, list(hidden_layers), num_steps, learning_rate, max_epochs)

    trained_models = []
    t0 = time.perf_counter()
    for target_idx in range(d):
        other_indices = [j for j in range(d) if j != target_idx]
        best_model, _, _ = trainer.train_model(
            training_data, target_idx, other_indices, regularizations, validation_data
        )
        trained_models.append(best_model)
    total_time = time.perf_counter() - t0
    return trained_models, total_time


def train_kan_models(
    training_data,
    validation_data,
    d,
    hidden_dim=4,
    grid_size=8,
    penalty_type="functional",
    regularizations=(1.0, 0.1, 0.01, 0.001, 0.0),
    learning_rate=0.01,
    max_epochs=50,
    dev="cpu",
):
    """Trains MonotonicKANMap models across all d features."""
    base_map = [
        MonotonicKANMap(d, target_index=k, hidden_dim=hidden_dim, grid_size=grid_size, penalty_type=penalty_type, dev=dev)
        for k in range(d)
    ]
    trainer = KANTrainer(
        base_map,
        d,
        hidden_dim=hidden_dim,
        grid_size=grid_size,
        penalty_type=penalty_type,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        patience=10,
        dev=dev,
    )

    trained_models = []
    t0 = time.perf_counter()
    for target_idx in range(d):
        other_indices = [j for j in range(d) if j != target_idx]
        best_model, _, _, _ = trainer.train_model(
            training_data, target_idx, other_indices, regularizations, validation_data
        )
        trained_models.append(best_model)
    total_time = time.perf_counter() - t0
    return trained_models, total_time


def evaluate_umnn_precision(trained_models, test_data, d):
    """Computes symmetric normalized precision matrix for UMNN models."""
    t0 = time.perf_counter()
    evaluator = PrecisionMatrixComputer(trained_models, test_data, d)
    raw_prec = evaluator.compute()
    sym_prec = evaluator.make_symmetric(raw_prec)
    eval_time = time.perf_counter() - t0
    return sym_prec, eval_time


def evaluate_kan_precision(trained_models, test_data, d):
    """Computes symmetric normalized precision matrix for KAN models."""
    t0 = time.perf_counter()
    comp = KANPrecisionComputer(trained_models, test_data, d)
    raw_prec = comp.compute(split=False)
    sym_prec = comp.make_symmetric(raw_prec)
    eval_time = time.perf_counter() - t0
    return sym_prec, eval_time


# -------------------------------------------------------------------------
# Benchmark Execution Functions
# -------------------------------------------------------------------------

def run_gaussian_benchmark(output_dir):
    """Runs Experiment 1: Gaussian Benchmark (d=10)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: GAUSSIAN BENCHMARK (d=10)")
    print("=" * 70)

    # Load pre-generated Gaussian data
    trn_data = torch.tensor(np.loadtxt(REPO_ROOT / "data/gauss/g_trn_5000_10d.txt", delimiter=","), dtype=torch.float32)
    val_data = torch.tensor(np.loadtxt(REPO_ROOT / "data/gauss/g_val_10000_10d.txt", delimiter=","), dtype=torch.float32)[:2000]
    tst_data = torch.tensor(np.loadtxt(REPO_ROOT / "data/gauss/g_tst_10000_10d.txt", delimiter=","), dtype=torch.float32)[:2000]

    d = 10
    true_adj = get_gaussian_ground_truth(d)
    regs = [1.0, 0.1, 0.01, 0.001, 0.0]

    results = {}

    # 1. UMNN
    print("Training UMNN baseline...")
    tracemalloc.start()
    umnn_models, umnn_train_time = train_umnn_models(
        trn_data, val_data, d, hidden_layers=[16, 32], num_steps=50, regularizations=regs, max_epochs=50
    )
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    umnn_mem_mb = peak / (1024 * 1024)

    print(f"UMNN training finished in {umnn_train_time:.2f} s. Evaluating precision matrix...")
    umnn_prec, umnn_eval_time = evaluate_umnn_precision(umnn_models, tst_data, d)
    umnn_metrics = compute_graph_metrics(umnn_prec, true_adj)

    results["UMNN"] = {
        "train_time_sec": umnn_train_time,
        "eval_time_sec": umnn_eval_time,
        "peak_mem_mb": umnn_mem_mb,
        "metrics": umnn_metrics,
        "precision_matrix": umnn_prec.tolist(),
    }

    # 2. MonotonicKAN (Functional Penalty)
    print("Training MonotonicKAN (Functional Penalty)...")
    tracemalloc.start()
    kan_func_models, kan_func_train_time = train_kan_models(
        trn_data, val_data, d, hidden_dim=4, grid_size=8, penalty_type="functional", regularizations=regs, max_epochs=50
    )
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    kan_func_mem_mb = peak / (1024 * 1024)

    print(f"KAN (Functional) training finished in {kan_func_train_time:.2f} s. Evaluating precision matrix...")
    kan_func_prec, kan_func_eval_time = evaluate_kan_precision(kan_func_models, tst_data, d)
    kan_func_metrics = compute_graph_metrics(kan_func_prec, true_adj)

    results["MonotonicKAN_Functional"] = {
        "train_time_sec": kan_func_train_time,
        "eval_time_sec": kan_func_eval_time,
        "peak_mem_mb": kan_func_mem_mb,
        "metrics": kan_func_metrics,
        "precision_matrix": kan_func_prec.tolist(),
    }

    # 3. MonotonicKAN (Group Lasso Penalty)
    print("Training MonotonicKAN (Group Lasso Penalty)...")
    tracemalloc.start()
    kan_grp_models, kan_grp_train_time = train_kan_models(
        trn_data, val_data, d, hidden_dim=4, grid_size=8, penalty_type="coeff_group", regularizations=regs, max_epochs=50
    )
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    kan_grp_mem_mb = peak / (1024 * 1024)

    print(f"KAN (Group Lasso) training finished in {kan_grp_train_time:.2f} s. Evaluating precision matrix...")
    kan_grp_prec, kan_grp_eval_time = evaluate_kan_precision(kan_grp_models, tst_data, d)
    kan_grp_metrics = compute_graph_metrics(kan_grp_prec, true_adj)

    results["MonotonicKAN_GroupLasso"] = {
        "train_time_sec": kan_grp_train_time,
        "eval_time_sec": kan_grp_eval_time,
        "peak_mem_mb": kan_grp_mem_mb,
        "metrics": kan_grp_metrics,
        "precision_matrix": kan_grp_prec.tolist(),
    }

    # Print summary table
    print("\n--- Gaussian Benchmark Results ---")
    print(f"{'Model':<25} | {'Train (s)':<10} | {'Eval (s)':<10} | {'Mem (MB)':<10} | {'tau=0.2 F1':<10} | {'tau=0.1 F1':<10} | {'tau=0.05 F1':<10}")
    print("-" * 100)
    for name, r in results.items():
        m = r["metrics"]
        print(f"{name:<25} | {r['train_time_sec']:<10.2f} | {r['eval_time_sec']:<10.3f} | {r['peak_mem_mb']:<10.1f} | {m[0.2]['f1']:<10.3f} | {m[0.1]['f1']:<10.3f} | {m[0.05]['f1']:<10.3f}")

    return results


def run_butterfly_benchmark_sample_sizes(output_dir):
    """
    Runs Experiment 2: Butterfly Benchmark (d=10) sample size sensitivity (M in {500, 1000, 5000}).
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: BUTTERFLY BENCHMARK (d=10) SAMPLE SIZE SENSITIVITY")
    print("=" * 70)

    # Load butterfly datasets
    full_trn = torch.tensor(np.loadtxt(REPO_ROOT / "data/butterfly/bf_trn_5000_10d.txt", delimiter=","), dtype=torch.float32)
    val_data = torch.tensor(np.loadtxt(REPO_ROOT / "data/butterfly/bf_val_10000_10d.txt", delimiter=","), dtype=torch.float32)[:2000]
    tst_data = torch.tensor(np.loadtxt(REPO_ROOT / "data/butterfly/bf_tst_10000_10d.txt", delimiter=","), dtype=torch.float32)[:2000]

    d = 10
    true_adj = get_butterfly_ground_truth(d)
    sample_sizes = [500, 1000, 5000]
    regs = [1.0, 0.1, 0.01, 0.001]

    results = {}

    for M in sample_sizes:
        print(f"\n>>> Running for M = {M} training samples...")
        trn_M = full_trn[:M]
        results[M] = {}

        # UMNN
        print(f"  Training UMNN (M={M})...")
        t0 = time.perf_counter()
        umnn_models, umnn_time = train_umnn_models(trn_M, val_data, d, hidden_layers=[16, 32], regularizations=regs, max_epochs=40)
        umnn_prec, umnn_eval_time = evaluate_umnn_precision(umnn_models, tst_data, d)
        umnn_m = compute_graph_metrics(umnn_prec, true_adj)
        results[M]["UMNN"] = {
            "train_time_sec": umnn_time,
            "eval_time_sec": umnn_eval_time,
            "metrics": umnn_m,
        }

        # MonotonicKAN
        print(f"  Training MonotonicKAN (M={M})...")
        kan_models, kan_time = train_kan_models(trn_M, val_data, d, hidden_dim=4, grid_size=8, penalty_type="functional", regularizations=regs, max_epochs=40)
        kan_prec, kan_eval_time = evaluate_kan_precision(kan_models, tst_data, d)
        kan_m = compute_graph_metrics(kan_prec, true_adj)
        results[M]["MonotonicKAN"] = {
            "train_time_sec": kan_time,
            "eval_time_sec": kan_eval_time,
            "metrics": kan_m,
        }

        # MonotonicKAN Group Lasso
        print(f"  Training MonotonicKAN Group Lasso (M={M})...")
        kan_g_models, kan_g_time = train_kan_models(trn_M, val_data, d, hidden_dim=4, grid_size=8, penalty_type="coeff_group", regularizations=regs, max_epochs=40)
        kan_g_prec, kan_g_eval_time = evaluate_kan_precision(kan_g_models, tst_data, d)
        kan_g_m = compute_graph_metrics(kan_g_prec, true_adj)
        results[M]["MonotonicKAN_GroupLasso"] = {
            "train_time_sec": kan_g_time,
            "eval_time_sec": kan_g_eval_time,
            "metrics": kan_g_m,
        }

    # Print summary table
    print("\n--- Butterfly Sample Size Sensitivity Results (tau = 0.1) ---")
    print(f"{'M':<6} | {'Model':<24} | {'Train (s)':<10} | {'Eval (s)':<10} | {'F1':<8} | {'FPR':<8} | {'Precision':<10} | {'Recall':<8}")
    print("-" * 95)
    for M in sample_sizes:
        for model_name, r in results[M].items():
            m1 = r["metrics"][0.1]
            print(f"{M:<6} | {model_name:<24} | {r['train_time_sec']:<10.2f} | {r['eval_time_sec']:<10.3f} | {m1['f1']:<8.3f} | {m1['fpr']:<8.3f} | {m1['precision']:<10.3f} | {m1['recall']:<8.3f}")

    return results


def run_butterfly_40d_benchmark(output_dir):
    """
    Runs Experiment 3: Butterfly Benchmark (d=40).
    d=40, r=20 pairs, 20 true edges out of 780 pairs.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: HIGH-DIMENSIONAL BUTTERFLY BENCHMARK (d=40)")
    print("=" * 70)

    trn_data = torch.tensor(np.loadtxt(REPO_ROOT / "data/butterfly/bf_trn_5000_40d.txt", delimiter=","), dtype=torch.float32)
    val_data = torch.tensor(np.loadtxt(REPO_ROOT / "data/butterfly/bf_val_10000_40d.txt", delimiter=","), dtype=torch.float32)[:2000]
    tst_data = torch.tensor(np.loadtxt(REPO_ROOT / "data/butterfly/bf_tst_10000_40d.txt", delimiter=","), dtype=torch.float32)[:2000]

    d = 40
    true_adj = get_butterfly_ground_truth(d)
    regs = [0.1, 0.01, 0.001]

    results = {}

    # 1. MonotonicKAN (Functional Penalty)
    print("Training MonotonicKAN (d=40, Functional Penalty)...")
    tracemalloc.start()
    kan_models, kan_train_time = train_kan_models(
        trn_data, val_data, d, hidden_dim=4, grid_size=8, penalty_type="functional", regularizations=regs, max_epochs=30
    )
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    kan_mem_mb = peak / (1024 * 1024)

    print(f"KAN d=40 trained in {kan_train_time:.2f} s. Computing precision matrix...")
    kan_prec, kan_eval_time = evaluate_kan_precision(kan_models, tst_data, d)
    kan_metrics = compute_graph_metrics(kan_prec, true_adj)

    results["MonotonicKAN_Functional"] = {
        "train_time_sec": kan_train_time,
        "eval_time_sec": kan_eval_time,
        "peak_mem_mb": kan_mem_mb,
        "metrics": kan_metrics,
        "precision_matrix": kan_prec.tolist(),
    }

    # 2. MonotonicKAN (Group Lasso Penalty)
    print("Training MonotonicKAN (d=40, Group Lasso Penalty)...")
    tracemalloc.start()
    kan_g_models, kan_g_train_time = train_kan_models(
        trn_data, val_data, d, hidden_dim=4, grid_size=8, penalty_type="coeff_group", regularizations=regs, max_epochs=30
    )
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    kan_g_mem_mb = peak / (1024 * 1024)

    print(f"KAN Group Lasso d=40 trained in {kan_g_train_time:.2f} s. Computing precision matrix...")
    kan_g_prec, kan_g_eval_time = evaluate_kan_precision(kan_g_models, tst_data, d)
    kan_g_metrics = compute_graph_metrics(kan_g_prec, true_adj)

    results["MonotonicKAN_GroupLasso"] = {
        "train_time_sec": kan_g_train_time,
        "eval_time_sec": kan_g_eval_time,
        "peak_mem_mb": kan_g_mem_mb,
        "metrics": kan_g_metrics,
        "precision_matrix": kan_g_prec.tolist(),
    }

    # 3. UMNN Baseline Benchmark on representative subset (to avoid 30+ min timeout)
    print("Benchmarking UMNN baseline for d=40 on sample features to measure per-variable cost...")
    d_sub = 4  # benchmark 4 features and project
    tracemalloc.start()
    t0 = time.perf_counter()
    umnn_sub_models = []
    base_map = [MonotonicNN(d, [16, 32], 50, "cpu") for _ in range(d_sub)]
    trainer = MapTrainer(base_map, d, [16, 32], 50, 0.01, 30)
    for k in range(d_sub):
        others = [j for j in range(d) if j != k]
        best_m, _, _ = trainer.train_model(trn_data, k, others, regs, val_data)
        umnn_sub_models.append(best_m)
    umnn_sub_time = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    umnn_mem_mb = peak / (1024 * 1024)

    umnn_projected_train_time = (umnn_sub_time / d_sub) * d

    # Evaluate UMNN 1 row precision to benchmark eval time for d=40
    t0 = time.perf_counter()
    ev = PrecisionMatrixComputer(umnn_sub_models[:1], tst_data[:500], d)
    _ = ev.compute()
    umnn_1row_eval_time = time.perf_counter() - t0
    umnn_projected_eval_time = (umnn_1row_eval_time * d) * (2000 / 500)

    results["UMNN_Projected"] = {
        "train_time_sec": umnn_projected_train_time,
        "eval_time_sec": umnn_projected_eval_time,
        "peak_mem_mb": umnn_mem_mb,
        "note": f"Measured on {d_sub} features ({umnn_sub_time:.1f} s) and scaled to {d} features",
    }

    print("\n--- High-Dimensional Butterfly (d=40) Results ---")
    print(f"{'Model':<25} | {'Train (s)':<10} | {'Eval (s)':<10} | {'Mem (MB)':<10} | {'tau=0.2 F1':<10} | {'tau=0.1 F1':<10} | {'tau=0.05 F1':<10}")
    print("-" * 100)
    for name, r in results.items():
        if "metrics" in r:
            m = r["metrics"]
            print(f"{name:<25} | {r['train_time_sec']:<10.2f} | {r['eval_time_sec']:<10.3f} | {r['peak_mem_mb']:<10.1f} | {m[0.2]['f1']:<10.3f} | {m[0.1]['f1']:<10.3f} | {m[0.05]['f1']:<10.3f}")
        else:
            print(f"{name:<25} | {r['train_time_sec']:<10.2f} | {r['eval_time_sec']:<10.3f} | {r['peak_mem_mb']:<10.1f} | {'N/A (Projected)':<10}")

    return results


def run_profiling_and_quadrature_breakdown():
    """
    Runs Experiment 4: Profiling & Quadrature Overhead.
    Compares:
    - Forward + backward training throughput (samples/sec)
    - Precision matrix evaluation speed:
      1) Analytic KAN
      2) torch.func vmap KAN
      3) Autograd UMNN (50-step quadrature)
    - Parameter count and model size
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: COMPUTATIONAL PROFILING & QUADRATURE OVERHEAD BREAKDOWN")
    print("=" * 70)

    d = 10
    batch_size = 1000
    x = torch.randn(batch_size, d)

    # 1. Parameter Count
    umnn = MonotonicNN(d, [16, 32], 50, "cpu")
    kan = MonotonicKANMap(d, target_index=0, hidden_dim=4, grid_size=8)

    umnn_params = sum(p.numel() for p in umnn.parameters())
    kan_params = sum(p.numel() for p in kan.parameters())

    print(f"Parameter Count: UMNN = {umnn_params}, MonotonicKAN = {kan_params}")

    # 2. Training Throughput (samples/sec)
    n_iters = 50

    # UMNN training throughput
    opt_u = optim.Adam(umnn.parameters(), lr=0.01)
    # Warmup
    for _ in range(5):
        target = x[:, [0]].detach().requires_grad_(True)
        h = x[:, 1:]
        out = umnn(target, h)
        jac = torch.autograd.grad(out, target, torch.ones_like(out), create_graph=True)[0]
        loss = (0.5 * out**2 - torch.log(jac)).mean()
        loss.backward()
        opt_u.step()
        opt_u.zero_grad()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        target = x[:, [0]].detach().requires_grad_(True)
        h = x[:, 1:]
        out = umnn(target, h)
        jac = torch.autograd.grad(out, target, torch.ones_like(out), create_graph=True)[0]
        loss = (0.5 * out**2 - torch.log(jac)).mean()
        loss.backward()
        opt_u.step()
        opt_u.zero_grad()
    t_umnn_train = time.perf_counter() - t0
    umnn_train_fps = (n_iters * batch_size) / t_umnn_train

    # KAN training throughput
    opt_k = optim.Adam(kan.parameters(), lr=0.01)
    for _ in range(5):
        loss, _, _ = kan.compute_loss(x, reg_value=0.01)
        loss.backward()
        opt_k.step()
        opt_k.zero_grad()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        loss, _, _ = kan.compute_loss(x, reg_value=0.01)
        loss.backward()
        opt_k.step()
        opt_k.zero_grad()
    t_kan_train = time.perf_counter() - t0
    kan_train_fps = (n_iters * batch_size) / t_kan_train

    # 3. Precision Matrix Evaluation Throughput
    N_eval = 1000
    eval_data = torch.randn(N_eval, d)

    # UMNN (1 row)
    ev_u = PrecisionMatrixComputer([umnn], eval_data, d)
    t0 = time.perf_counter()
    _ = ev_u.compute()
    t_umnn_prec_1row = time.perf_counter() - t0
    t_umnn_prec_full = t_umnn_prec_1row * d

    # KAN Analytic (full 10 rows)
    kan_models = [MonotonicKANMap(d, target_index=k, hidden_dim=4, grid_size=8) for k in range(d)]
    comp_k = KANPrecisionComputer(kan_models, eval_data, d)
    t0 = time.perf_counter()
    _ = comp_k.compute()
    t_kan_prec_full = time.perf_counter() - t0

    # KAN torch.func (1 row)
    t0 = time.perf_counter()
    _ = kan.compute_precision_row_torch_func(eval_data[:200])  # evaluate on 200 samples to keep fast
    t_vmap_sample = (time.perf_counter() - t0) * (N_eval / 200.0)
    t_vmap_prec_full = t_vmap_sample * d

    print("\n--- Profiling Summary ---")
    print(f"Training Throughput (Batch={batch_size}):")
    print(f"  UMNN:         {umnn_train_fps:10.1f} samples/sec ({t_umnn_train/n_iters*1000:.2f} ms/epoch)")
    print(f"  MonotonicKAN: {kan_train_fps:10.1f} samples/sec ({t_kan_train/n_iters*1000:.2f} ms/epoch)")
    print(f"  Speedup:      {kan_train_fps / umnn_train_fps:.2f}x")

    print(f"\nGeneralized Precision Matrix Extraction Speed (N={N_eval}, d={d}):")
    print(f"  UMNN (Quadrature + Autograd):  {t_umnn_prec_full:10.3f} s")
    print(f"  KAN (torch.func vmap):         {t_vmap_prec_full:10.3f} s")
    print(f"  KAN (Exact Analytic Chain):    {t_kan_prec_full:10.3f} s")
    print(f"  Analytic vs UMNN Speedup:      {t_umnn_prec_full / t_kan_prec_full:.1f}x")
    print(f"  Analytic vs torch.func Speedup: {t_vmap_prec_full / t_kan_prec_full:.1f}x")

    profiling_results = {
        "umnn_params": umnn_params,
        "kan_params": kan_params,
        "training_throughput": {
            "umnn_samples_per_sec": umnn_train_fps,
            "kan_samples_per_sec": kan_train_fps,
            "training_speedup": kan_train_fps / umnn_train_fps,
        },
        "precision_evaluation_seconds": {
            "umnn_quadrature": t_umnn_prec_full,
            "kan_torch_func": t_vmap_prec_full,
            "kan_analytic": t_kan_prec_full,
            "speedup_vs_umnn": t_umnn_prec_full / t_kan_prec_full,
            "speedup_vs_torch_func": t_vmap_prec_full / t_kan_prec_full,
        },
    }
    return profiling_results




def run_mixed_coupling_benchmark(output_dir):
    """
    Runs Experiment 5: 6-Dimensional Mixed Additive & Multiplicative Coupling Graph.
    Demonstrates location-scale decoupling in ScaleModulatedKANMap vs Additive MonotonicKANMap.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: MIXED ADDITIVE & MULTIPLICATIVE COUPLING BENCHMARK (d=6)")
    print("=" * 70)

    n_trn, n_val, n_tst = 2000, 800, 1000
    full_data = generate_mixed_coupling_dataset(n_trn + n_val + n_tst, standardize=True, seed=42)
    trn_data = full_data[:n_trn]
    val_data = full_data[n_trn : n_trn + n_val]
    tst_data = full_data[n_trn + n_val :]

    d = 6
    true_adj = get_mixed_coupling_ground_truth()
    regs = [0.01, 0.005]
    max_epochs = 30

    # 1. Additive KAN
    print("Training Additive MonotonicKANMap...")
    t0 = time.perf_counter()
    base_add = [MonotonicKANMap(d, target_index=k, hidden_dim=4, grid_size=8, conditioning_scale=False) for k in range(d)]
    trainer_add = KANTrainer(base_add, d, hidden_dim=4, grid_size=8, max_epochs=max_epochs)
    add_models = []
    for k in range(d):
        others = [j for j in range(d) if j != k]
        best_m, _, _, _ = trainer_add.train_model(trn_data, k, others, regs, val_data)
        add_models.append(best_m)
    add_time = time.perf_counter() - t0

    comp_add = KANPrecisionComputer(add_models, tst_data, d)
    raw_add_asym = comp_add.compute()
    raw_add = 0.5 * (raw_add_asym + raw_add_asym.T)
    np.fill_diagonal(raw_add, 1.0)
    norm_add = comp_add.make_symmetric(raw_add_asym)

    # 2. Scale-Modulated KAN
    print("Training ScaleModulatedKANMap...")
    t0 = time.perf_counter()
    base_sm = [ScaleModulatedKANMap(d, target_index=k, hidden_dim=4, grid_size=8) for k in range(d)]
    trainer_sm = KANTrainer(base_sm, d, hidden_dim=4, grid_size=8, max_epochs=max_epochs)
    sm_models = []
    for k in range(d):
        others = [j for j in range(d) if j != k]
        best_m, _, _, _ = trainer_sm.train_model(trn_data, k, others, regs, val_data)
        sm_models.append(best_m)
    sm_time = time.perf_counter() - t0

    comp_sm = KANPrecisionComputer(sm_models, tst_data, d)
    raw_sm_asym = comp_sm.compute()
    raw_sm = 0.5 * (raw_sm_asym + raw_sm_asym.T)
    np.fill_diagonal(raw_sm, 1.0)
    norm_sm = comp_sm.make_symmetric(raw_sm_asym)

    results = {
        "ground_truth_adj": true_adj.tolist(),
        "AdditiveMonotonicKAN": {
            "train_time_sec": add_time,
            "raw_precision_matrix": raw_add.tolist(),
            "normalized_precision_matrix": norm_add.tolist(),
        },
        "ScaleModulatedKAN": {
            "train_time_sec": sm_time,
            "raw_precision_matrix": raw_sm.tolist(),
            "normalized_precision_matrix": norm_sm.tolist(),
        },
    }

    out_json = output_dir / "mixed_coupling_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print("\n--- Mixed Coupling Symmetrized Precision Entries ---")
    print(f"  Purely Additive Edge (0, 1):       Additive KAN = {raw_add[0, 1]:.3f} | SM-KAN = {raw_sm[0, 1]:.3f}")
    print(f"  Purely Multiplicative Edge (2, 3): Additive KAN = {raw_add[2, 3]:.3f} | SM-KAN = {raw_sm[2, 3]:.3f}")
    print(f"  Mixed Additive Edge (4, 5):        Additive KAN = {raw_add[4, 5]:.3f} | SM-KAN = {raw_sm[4, 5]:.3f}")
    print(f"  Mixed Multiplicative Edge (2, 5):  Additive KAN = {raw_add[2, 5]:.3f} | SM-KAN = {raw_sm[2, 5]:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive L-SING Experiment Suite: Monotonic KAN Transport Maps vs UMNN"
    )
    parser.add_argument("--verify-theory", action="store_true", help="Run live verification of Theorems 4.4, 5.3, and derivative exactness")
    parser.add_argument("--quick", action="store_true", help="Run quick profiling, theory check, and mixed coupling demo")
    parser.add_argument("--mixed-coupling", action="store_true", help="Run 6D mixed additive/multiplicative coupling benchmark")
    parser.add_argument("--generate-plots", action="store_true", help="Generate publication-ready figures in results/kan_benchmarks/figures")
    parser.add_argument("--experiment", type=str, choices=["all", "profiling", "gaussian", "butterfly", "butterfly40d", "mixed", "theory"], default="all")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    output_dir = REPO_ROOT / "results/kan_benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("L-SING EXPERIMENTAL SUITE: MONOTONIC KAN TRANSPORT MAPS")
    print("=" * 80)

    # 1. Theory Verification Mode
    if args.verify_theory or args.experiment == "theory":
        run_all_theory_verifications()
        return

    # 2. Quick Mode
    if args.quick:
        print("\n>>> RUNNING QUICK DEMO SUITE (Theory Verification + Profiling + Mixed Coupling)...")
        run_all_theory_verifications()
        prof_res = run_profiling_and_quadrature_breakdown()
        mixed_res = run_mixed_coupling_benchmark(output_dir)
        try:
            import generate_benchmark_plots
            generate_benchmark_plots.main()
        except Exception as e:
            print(f"Plot generation note: {e}")
        print("\nQuick demo suite finished successfully!")
        return

    # 3. Targeted or Full Experiments
    all_results = {}

    if args.experiment in ["all", "profiling"]:
        all_results["profiling"] = run_profiling_and_quadrature_breakdown()

    if args.experiment in ["all", "gaussian"]:
        all_results["gaussian_d10"] = run_gaussian_benchmark(output_dir)

    if args.experiment in ["all", "butterfly"]:
        all_results["butterfly_sample_size_sensitivity"] = run_butterfly_benchmark_sample_sizes(output_dir)

    if args.experiment in ["all", "butterfly40d"]:
        all_results["butterfly_d40"] = run_butterfly_40d_benchmark(output_dir)

    if args.experiment in ["all", "mixed"] or args.mixed_coupling:
        all_results["mixed_coupling_d6"] = run_mixed_coupling_benchmark(output_dir)

    # Save summary JSON
    json_path = output_dir / "benchmark_raw_results.json"
    if all_results:
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nBenchmark results successfully saved to {json_path}")

    # Generate figures if requested
    if args.generate_plots or args.experiment == "all":
        try:
            import generate_benchmark_plots
            generate_benchmark_plots.main()
        except Exception as e:
            print(f"Could not auto-generate plots: {e}")


if __name__ == '__main__':
    main()
