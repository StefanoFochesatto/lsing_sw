#!/usr/bin/env python3
"""
Gaussian Benchmark (d=10): UMNN vs MonotonicKAN (Functional & Group Lasso).
Measures:
- Edge recovery: F1, FPR, Precision, Recall at tau in {0.2, 0.1, 0.05}
- Training wall-clock time
- Peak memory consumption
- Precision matrix evaluation speed
"""

import sys
import time
import json
import tracemalloc
import numpy as np
import torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from models.UMNN import MonotonicNN
from models.kan_transport import MonotonicKANMap
from trainers.map_trainer import MapTrainer
from trainers.kan_trainer import KANTrainer, KANPrecisionComputer
from evaluators.precision_matrix_computer import PrecisionMatrixComputer


def get_gaussian_ground_truth(d=10):
    adj = np.zeros((d, d), dtype=int)
    edges = [(0, 4), (2, 6), (5, 7), (6, 9)]
    for u, v in edges:
        adj[u, v] = adj[v, u] = 1
    return adj


def compute_graph_metrics(precision_matrix, true_adj, thresholds=(0.2, 0.1, 0.05)):
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

        results[str(tau)] = {
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


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    output_dir = REPO_ROOT / "results/kan_benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GAUSSIAN BENCHMARK (d=10)")
    print("=" * 70)

    trn_data = torch.tensor(np.loadtxt(REPO_ROOT / "data/gauss/g_trn_5000_10d.txt", delimiter=","), dtype=torch.float32)
    val_data = torch.tensor(np.loadtxt(REPO_ROOT / "data/gauss/g_val_10000_10d.txt", delimiter=","), dtype=torch.float32)[:2000]
    tst_data = torch.tensor(np.loadtxt(REPO_ROOT / "data/gauss/g_tst_10000_10d.txt", delimiter=","), dtype=torch.float32)[:2000]

    d = 10
    true_adj = get_gaussian_ground_truth(d)
    regs = [0.1, 0.01, 0.001, 0.0]
    max_epochs = 35

    results = {}

    # 1. UMNN Baseline
    print("1/3: Training UMNN...")
    tracemalloc.start()
    t0 = time.perf_counter()
    base_map = [MonotonicNN(d, [16, 32], 50, "cpu") for _ in range(d)]
    trainer = MapTrainer(base_map, d, [16, 32], 50, 0.01, max_epochs)
    umnn_models = []
    for k in range(d):
        others = [j for j in range(d) if j != k]
        best_m, _, _ = trainer.train_model(trn_data, k, others, regs, val_data)
        umnn_models.append(best_m)
    umnn_train_time = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    umnn_mem_mb = peak / (1024 * 1024)

    t0 = time.perf_counter()
    ev = PrecisionMatrixComputer(umnn_models, tst_data, d)
    umnn_prec = ev.make_symmetric(ev.compute())
    umnn_eval_time = time.perf_counter() - t0
    umnn_metrics = compute_graph_metrics(umnn_prec, true_adj)

    results["UMNN"] = {
        "train_time_sec": umnn_train_time,
        "eval_time_sec": umnn_eval_time,
        "peak_mem_mb": umnn_mem_mb,
        "metrics": umnn_metrics,
        "precision_matrix": umnn_prec.tolist(),
    }
    print(f"   UMNN Done: Train={umnn_train_time:.1f}s, Eval={umnn_eval_time:.2f}s, tau=0.1 F1={umnn_metrics['0.1']['f1']:.3f}")

    # 2. MonotonicKAN (Functional Penalty)
    print("2/3: Training MonotonicKAN (Functional Penalty)...")
    tracemalloc.start()
    t0 = time.perf_counter()
    base_map_kan_func = [MonotonicKANMap(d, target_index=k, hidden_dim=4, grid_size=8, penalty_type="functional") for k in range(d)]
    kan_trainer_func = KANTrainer(base_map_kan_func, d, hidden_dim=4, grid_size=8, penalty_type="functional", max_epochs=max_epochs)
    kan_func_models = []
    for k in range(d):
        others = [j for j in range(d) if j != k]
        best_m, _, _, _ = kan_trainer_func.train_model(trn_data, k, others, regs, val_data)
        kan_func_models.append(best_m)
    kan_func_train_time = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    kan_func_mem_mb = peak / (1024 * 1024)

    t0 = time.perf_counter()
    comp = KANPrecisionComputer(kan_func_models, tst_data, d)
    kan_func_prec = comp.make_symmetric(comp.compute())
    kan_func_eval_time = time.perf_counter() - t0
    kan_func_metrics = compute_graph_metrics(kan_func_prec, true_adj)

    results["MonotonicKAN_Functional"] = {
        "train_time_sec": kan_func_train_time,
        "eval_time_sec": kan_func_eval_time,
        "peak_mem_mb": kan_func_mem_mb,
        "metrics": kan_func_metrics,
        "precision_matrix": kan_func_prec.tolist(),
    }
    print(f"   KAN Functional Done: Train={kan_func_train_time:.1f}s, Eval={kan_func_eval_time:.2f}s, tau=0.1 F1={kan_func_metrics['0.1']['f1']:.3f}")

    # 3. MonotonicKAN (Group Lasso Penalty)
    print("3/3: Training MonotonicKAN (Group Lasso Penalty)...")
    tracemalloc.start()
    t0 = time.perf_counter()
    base_map_kan_grp = [MonotonicKANMap(d, target_index=k, hidden_dim=4, grid_size=8, penalty_type="coeff_group") for k in range(d)]
    kan_trainer_grp = KANTrainer(base_map_kan_grp, d, hidden_dim=4, grid_size=8, penalty_type="coeff_group", max_epochs=max_epochs)
    kan_grp_models = []
    for k in range(d):
        others = [j for j in range(d) if j != k]
        best_m, _, _, _ = kan_trainer_grp.train_model(trn_data, k, others, regs, val_data)
        kan_grp_models.append(best_m)
    kan_grp_train_time = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    kan_grp_mem_mb = peak / (1024 * 1024)

    t0 = time.perf_counter()
    comp_grp = KANPrecisionComputer(kan_grp_models, tst_data, d)
    kan_grp_prec = comp_grp.make_symmetric(comp_grp.compute())
    kan_grp_eval_time = time.perf_counter() - t0
    kan_grp_metrics = compute_graph_metrics(kan_grp_prec, true_adj)

    results["MonotonicKAN_GroupLasso"] = {
        "train_time_sec": kan_grp_train_time,
        "eval_time_sec": kan_grp_eval_time,
        "peak_mem_mb": kan_grp_mem_mb,
        "metrics": kan_grp_metrics,
        "precision_matrix": kan_grp_prec.tolist(),
    }
    print(f"   KAN Group Lasso Done: Train={kan_grp_train_time:.1f}s, Eval={kan_grp_eval_time:.2f}s, tau=0.1 F1={kan_grp_metrics['0.1']['f1']:.3f}")

    # Save output JSON
    out_json = output_dir / "gaussian_benchmark.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved Gaussian benchmark results to {out_json}")


if __name__ == "__main__":
    main()
