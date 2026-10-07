#!/usr/bin/env python3
"""
Benchmark: Scale-Modulated Monotonic KAN vs Additive Monotonic KAN vs UMNN
on the 10-Dimensional Butterfly Distribution.
"""

import sys
import time
import json
import numpy as np
import torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from models.UMNN import MonotonicNN
from models.kan_transport import MonotonicKANMap, ScaleModulatedKANMap
from trainers.map_trainer import MapTrainer
from trainers.kan_trainer import KANTrainer, KANPrecisionComputer
from evaluators.precision_matrix_computer import PrecisionMatrixComputer


def get_butterfly_ground_truth(d=10):
    adj = np.zeros((d, d), dtype=int)
    num_pairs = d // 2
    for i in range(num_pairs):
        u, v = 2 * i, 2 * i + 1
        adj[u, v] = adj[v, u] = 1
    return adj


def compute_graph_metrics(precision_matrix, true_adj, thresholds=(0.3, 0.2, 0.1, 0.05)):
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
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": float(prec), "recall": float(rec), "f1": float(f1), "fpr": float(fpr)
        }
    return results


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    output_dir = REPO_ROOT / "results/kan_benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SCALE-MODULATED MONOTONIC KAN BUTTERFLY BENCHMARK (d=10)")
    print("=" * 80)

    full_trn = torch.tensor(np.loadtxt(REPO_ROOT / "data/butterfly/bf_trn_5000_10d.txt", delimiter=","), dtype=torch.float32)
    val_data = torch.tensor(np.loadtxt(REPO_ROOT / "data/butterfly/bf_val_10000_10d.txt", delimiter=","), dtype=torch.float32)[:2000]
    tst_data = torch.tensor(np.loadtxt(REPO_ROOT / "data/butterfly/bf_tst_10000_10d.txt", delimiter=","), dtype=torch.float32)[:2000]

    d = 10
    true_adj = get_butterfly_ground_truth(d)
    sample_sizes = [1000, 5000]
    regs = [0.01, 0.005]
    max_epochs = 35

    results = {}

    for M in sample_sizes:
        print(f"\n>>> Running for Sample Size M = {M}...")
        trn_M = full_trn[:M]
        results[str(M)] = {}

        # 1. Scale-Modulated Monotonic KAN
        print(f"  Training ScaleModulatedKANMap (M={M})...")
        base_map_sm = [ScaleModulatedKANMap(d, target_index=k, hidden_dim=4, grid_size=8) for k in range(d)]
        trainer_sm = KANTrainer(base_map_sm, d, hidden_dim=4, grid_size=8, max_epochs=max_epochs)
        t0 = time.perf_counter()
        sm_models = []
        for k in range(d):
            others = [j for j in range(d) if j != k]
            best_m, _, _, _ = trainer_sm.train_model(trn_M, k, others, regs, val_data)
            sm_models.append(best_m)
        sm_train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        comp_sm = KANPrecisionComputer(sm_models, tst_data, d)
        sm_prec = comp_sm.make_symmetric(comp_sm.compute())
        sm_eval_time = time.perf_counter() - t0
        sm_metrics = compute_graph_metrics(sm_prec, true_adj)

        results[str(M)]["ScaleModulatedKAN"] = {
            "train_time_sec": sm_train_time,
            "eval_time_sec": sm_eval_time,
            "metrics": sm_metrics,
            "precision_matrix": sm_prec.tolist()
        }
        print(f"    ScaleModulatedKAN (M={M}): Train={sm_train_time:.1f}s, Eval={sm_eval_time:.3f}s, tau=0.2 F1={sm_metrics['0.2']['f1']:.3f}, tau=0.1 F1={sm_metrics['0.1']['f1']:.3f}")

        # 2. Additive Monotonic KAN (Baseline)
        print(f"  Training Additive MonotonicKANMap (M={M})...")
        base_map_add = [MonotonicKANMap(d, target_index=k, hidden_dim=4, grid_size=8, conditioning_scale=False) for k in range(d)]
        trainer_add = KANTrainer(base_map_add, d, hidden_dim=4, grid_size=8, max_epochs=max_epochs)
        t0 = time.perf_counter()
        add_models = []
        for k in range(d):
            others = [j for j in range(d) if j != k]
            best_m, _, _, _ = trainer_add.train_model(trn_M, k, others, regs, val_data)
            add_models.append(best_m)
        add_train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        comp_add = KANPrecisionComputer(add_models, tst_data, d)
        add_prec = comp_add.make_symmetric(comp_add.compute())
        add_eval_time = time.perf_counter() - t0
        add_metrics = compute_graph_metrics(add_prec, true_adj)

        results[str(M)]["AdditiveMonotonicKAN"] = {
            "train_time_sec": add_train_time,
            "eval_time_sec": add_eval_time,
            "metrics": add_metrics,
            "precision_matrix": add_prec.tolist()
        }
        print(f"    AdditiveMonotonicKAN (M={M}): Train={add_train_time:.1f}s, Eval={add_eval_time:.3f}s, tau=0.2 F1={add_metrics['0.2']['f1']:.3f}, tau=0.1 F1={add_metrics['0.1']['f1']:.3f}")

    out_json = output_dir / "scale_modulated_butterfly_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {out_json}")


if __name__ == "__main__":
    main()
