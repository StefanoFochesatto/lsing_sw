#!/usr/bin/env python3
"""
Sparsity Penalty Ablation:
Compares:
1. L-SING Functional Penalty (Equation 5)
2. KAN Native Coefficient Group Lasso Penalty
3. Target Jacobian Penalty (Legacy MapTrainer line 71)

Evaluates:
- Training wall-clock time
- Final structural sparsity of conditioning weights
- Sensitivity / False Positive suppression (F1, FPR)
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

from models.kan_transport import MonotonicKANMap
from trainers.kan_trainer import KANTrainer, KANPrecisionComputer


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
    print("SPARSITY PENALTY ABLATION STUDY")
    print("=" * 70)

    trn_data = torch.tensor(np.loadtxt(REPO_ROOT / "data/gauss/g_trn_5000_10d.txt", delimiter=","), dtype=torch.float32)
    val_data = torch.tensor(np.loadtxt(REPO_ROOT / "data/gauss/g_val_10000_10d.txt", delimiter=","), dtype=torch.float32)[:2000]
    tst_data = torch.tensor(np.loadtxt(REPO_ROOT / "data/gauss/g_tst_10000_10d.txt", delimiter=","), dtype=torch.float32)[:2000]

    d = 10
    true_adj = get_gaussian_ground_truth(d)
    regs = [0.1, 0.01, 0.001, 0.0]
    max_epochs = 35

    penalties = ["functional", "coeff_group", "target_jacobian"]
    results = {}

    for p_type in penalties:
        print(f"\nEvaluating penalty_type = '{p_type}'...")
        t0 = time.perf_counter()
        base_map = [MonotonicKANMap(d, target_index=k, hidden_dim=4, grid_size=8, penalty_type=p_type) for k in range(d)]
        trainer = KANTrainer(base_map, d, hidden_dim=4, grid_size=8, penalty_type=p_type, max_epochs=max_epochs)
        models = []
        for k in range(d):
            others = [j for j in range(d) if j != k]
            best_m, _, _, _ = trainer.train_model(trn_data, k, others, regs, val_data)
            models.append(best_m)
        train_time = time.perf_counter() - t0

        # Measure weight sparsity
        cond_weight_l2_sum = 0.0
        for m in models:
            if m.inner_cond_weights is not None:
                cond_weight_l2_sum += float(torch.norm(m.inner_cond_weights).item())

        t0 = time.perf_counter()
        comp = KANPrecisionComputer(models, tst_data, d)
        prec = comp.make_symmetric(comp.compute())
        eval_time = time.perf_counter() - t0
        metrics = compute_graph_metrics(prec, true_adj)

        results[p_type] = {
            "train_time_sec": train_time,
            "eval_time_sec": eval_time,
            "cond_weight_l2_sum": cond_weight_l2_sum,
            "metrics": metrics,
            "precision_matrix": prec.tolist(),
        }
        print(f"  {p_type}: Train={train_time:.1f}s, CondWeightNorm={cond_weight_l2_sum:.3f}, tau=0.1 F1={metrics['0.1']['f1']:.3f}, FPR={metrics['0.1']['fpr']:.3f}")

    out_json = output_dir / "ablation_penalties.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved ablation results to {out_json}")


if __name__ == "__main__":
    main()
