#!/usr/bin/env python3
"""
High-Dimensional Butterfly Benchmark (d=40):
Evaluates scalability of MonotonicKAN on d=40 (20 true edges among 780 pairs).
Benchmarks:
- MonotonicKAN (Functional penalty)
- UMNN baseline (measured per-variable cost + projected 40-variable time)
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
from models.monotonic_kan import MonotonicKANMap
from trainers.map_trainer import MapTrainer
from trainers.kan_trainer import KANTrainer, KANPrecisionComputer
from evaluators.precision_matrix_computer import PrecisionMatrixComputer


def get_butterfly_ground_truth(d=40):
    adj = np.zeros((d, d), dtype=int)
    num_pairs = d // 2
    for i in range(num_pairs):
        u, v = 2 * i, 2 * i + 1
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
    print("HIGH-DIMENSIONAL BUTTERFLY BENCHMARK (d=40)")
    print("=" * 70)

    trn_data = torch.tensor(np.loadtxt(REPO_ROOT / "data/butterfly/bf_trn_5000_40d.txt", delimiter=","), dtype=torch.float32)
    val_data = torch.tensor(np.loadtxt(REPO_ROOT / "data/butterfly/bf_val_10000_40d.txt", delimiter=","), dtype=torch.float32)[:1000]
    tst_data = torch.tensor(np.loadtxt(REPO_ROOT / "data/butterfly/bf_tst_10000_40d.txt", delimiter=","), dtype=torch.float32)[:1000]

    d = 40
    true_adj = get_butterfly_ground_truth(d)
    regs = [0.05, 0.01]
    max_epochs = 20

    results = {}

    # 1. MonotonicKAN (Functional Penalty)
    print("1/2: Training MonotonicKAN d=40 (Functional Penalty)...")
    tracemalloc.start()
    t0 = time.perf_counter()
    base_map = [MonotonicKANMap(d, target_index=k, hidden_dim=4, grid_size=8, penalty_type="functional") for k in range(d)]
    trainer = KANTrainer(base_map, d, hidden_dim=4, grid_size=8, penalty_type="functional", max_epochs=max_epochs)
    kan_models = []
    for k in range(d):
        others = [j for j in range(d) if j != k]
        best_m, _, _, _ = trainer.train_model(trn_data, k, others, regs, val_data)
        kan_models.append(best_m)
    kan_train_time = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    kan_mem_mb = peak / (1024 * 1024)

    t0 = time.perf_counter()
    comp = KANPrecisionComputer(kan_models, tst_data, d)
    kan_prec = comp.make_symmetric(comp.compute())
    kan_eval_time = time.perf_counter() - t0
    kan_metrics = compute_graph_metrics(kan_prec, true_adj)

    results["MonotonicKAN"] = {
        "penalty_type": "functional",

        "train_time_sec": kan_train_time,
        "eval_time_sec": kan_eval_time,
        "peak_mem_mb": kan_mem_mb,
        "train_time_per_var_sec": kan_train_time / d,
        "eval_time_per_var_sec": kan_eval_time / d,
        "metrics": kan_metrics,
        "precision_matrix": kan_prec.tolist(),
    }
    print(f"   KAN Functional d=40 Done: Train={kan_train_time:.1f}s ({kan_train_time/d:.2f}s/var), Eval={kan_eval_time:.2f}s, tau=0.1 F1={kan_metrics['0.1']['f1']:.3f}, FPR={kan_metrics['0.1']['fpr']:.4f}")

    # 3. Benchmark UMNN per-variable cost on d=40
    print("2/2: Measuring UMNN per-variable baseline cost on d=40...")
    tracemalloc.start()
    t0 = time.perf_counter()
    base_map_sub = [MonotonicNN(d, [16, 32], 50, "cpu")]
    trainer_sub = MapTrainer(base_map_sub, d, [16, 32], 50, 0.01, 10)
    others = [j for j in range(d) if j != 0]
    best_m, _, _ = trainer_sub.train_model(trn_data, 0, others, regs, val_data)
    umnn_1var_train_time = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    umnn_mem_mb = peak / (1024 * 1024)

    # Scale 10 epochs to max_epochs (20)
    umnn_1var_train_time_scaled = umnn_1var_train_time * (max_epochs / 10.0)
    umnn_projected_train = umnn_1var_train_time_scaled * d

    t0 = time.perf_counter()
    ev_sub = PrecisionMatrixComputer([best_m], tst_data[:200], d)
    _ = ev_sub.compute()
    umnn_1row_eval_time = (time.perf_counter() - t0) * (1000.0 / 200.0)
    umnn_projected_eval = umnn_1row_eval_time * d

    results["UMNN_Projected"] = {
        "train_time_sec": umnn_projected_train,
        "eval_time_sec": umnn_projected_eval,
        "peak_mem_mb": umnn_mem_mb,
        "train_time_per_var_sec": umnn_1var_train_time_scaled,
        "eval_time_per_var_sec": umnn_1row_eval_time,
        "note": f"Measured on 1 variable ({umnn_1var_train_time:.1f} s for 10 epochs) and scaled to {d} variables and {max_epochs} epochs",
    }
    print(f"   UMNN Projected: Train={umnn_projected_train:.1f}s ({umnn_1var_train_time_scaled:.1f}s/var), Eval={umnn_projected_eval:.1f}s ({umnn_1row_eval_time:.1f}s/row)")

    out_json = output_dir / "butterfly_40d.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved Butterfly d=40 results to {out_json}")


if __name__ == "__main__":
    main()
