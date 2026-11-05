#!/usr/bin/env python3
"""
Butterfly Benchmark (d=10): UMNN Baseline vs MonotonicKAN (Functional Regularization)
across sample sizes M in {500, 1000, 5000}.
"""

import sys
import time
import json
import argparse
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


def get_butterfly_ground_truth(d=10):
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
    parser = argparse.ArgumentParser(description="Butterfly Benchmark Sample Size Sensitivity: UMNN vs MonotonicKAN")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer sample sizes and epochs)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dev_str = args.device
    if dev_str == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        dev_str = "cpu"

    output_dir = REPO_ROOT / "results/kan_benchmarks" / ("quick" if args.quick else "")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"BUTTERFLY BENCHMARK (d=10) SAMPLE SIZE SENSITIVITY [Device: {dev_str.upper()}]")
    print("=" * 70)

    full_trn = torch.tensor(np.loadtxt(REPO_ROOT / "data/butterfly/bf_trn_5000_10d.txt", delimiter=","), dtype=torch.float32)
    val_data = torch.tensor(np.loadtxt(REPO_ROOT / "data/butterfly/bf_val_10000_10d.txt", delimiter=","), dtype=torch.float32)[:2000]
    tst_data = torch.tensor(np.loadtxt(REPO_ROOT / "data/butterfly/bf_tst_10000_10d.txt", delimiter=","), dtype=torch.float32)[:2000]

    d = 10
    true_adj = get_butterfly_ground_truth(d)

    if args.quick:
        sample_sizes = [500, 1000]
        regs = [0.01]
        max_epochs = 5
        val_data = val_data[:200]
        tst_data = tst_data[:200]
        print(f"QUICK MODE: sample_sizes={sample_sizes}, regs=[0.01], max_epochs=5")
    else:
        sample_sizes = [500, 1000, 5000]
        regs = [0.1, 0.01, 0.001]
        max_epochs = 30

    results = {}

    for M in sample_sizes:
        print(f"\n--- Testing Sample Size M = {M} ---")
        trn_M = full_trn[:M]
        results[str(M)] = {}

        # 1. UMNN Baseline
        print(f"  [M={M}] Training UMNN...")
        t0 = time.perf_counter()
        base_map = [MonotonicNN(d, [16, 32], 50, dev=dev_str) for _ in range(d)]
        trainer = MapTrainer(base_map, d, [16, 32], 50, 0.01, max_epochs, dev=dev_str)
        umnn_models = []
        for k in range(d):
            others = [j for j in range(d) if j != k]
            best_m, _, _ = trainer.train_model(trn_M, k, others, regs, val_data)
            umnn_models.append(best_m)
        umnn_train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        ev = PrecisionMatrixComputer(umnn_models, tst_data, d, dev=dev_str)
        umnn_prec = ev.make_symmetric(ev.compute())
        umnn_eval_time = time.perf_counter() - t0
        umnn_metrics = compute_graph_metrics(umnn_prec, true_adj)

        results[str(M)]["UMNN"] = {
            "train_time_sec": umnn_train_time,
            "eval_time_sec": umnn_eval_time,
            "metrics": umnn_metrics,
            "precision_matrix": umnn_prec.tolist(),
        }
        print(f"    UMNN (M={M}): Train={umnn_train_time:.1f}s, Eval={umnn_eval_time:.2f}s, tau=0.1 F1={umnn_metrics['0.1']['f1']:.3f}, FPR={umnn_metrics['0.1']['fpr']:.3f}")

        # 2. MonotonicKAN (Functional Penalty)
        print(f"  [M={M}] Training MonotonicKAN (Functional)...")
        t0 = time.perf_counter()
        base_map_kan = [MonotonicKANMap(d, target_index=k, hidden_dim=4, grid_size=8, penalty_type="functional", dev=dev_str) for k in range(d)]
        kan_trainer = KANTrainer(base_map_kan, d, hidden_dim=4, grid_size=8, penalty_type="functional", max_epochs=max_epochs, dev=dev_str)
        kan_models = []
        for k in range(d):
            others = [j for j in range(d) if j != k]
            best_m, _, _, _ = kan_trainer.train_model(trn_M, k, others, regs, val_data)
            kan_models.append(best_m)
        kan_train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        comp = KANPrecisionComputer(kan_models, tst_data.to(dev_str), d)
        kan_prec = comp.make_symmetric(comp.compute())
        kan_eval_time = time.perf_counter() - t0
        kan_metrics = compute_graph_metrics(kan_prec, true_adj)

        results[str(M)]["MonotonicKAN_Functional"] = {
            "penalty_type": "functional",
            "train_time_sec": kan_train_time,
            "eval_time_sec": kan_eval_time,
            "metrics": kan_metrics,
            "precision_matrix": kan_prec.tolist(),
        }
        print(f"    KAN Func (M={M}): Train={kan_train_time:.1f}s, Eval={kan_eval_time:.2f}s, tau=0.1 F1={kan_metrics['0.1']['f1']:.3f}, FPR={kan_metrics['0.1']['fpr']:.3f}")

        # Save after each M
        out_json = output_dir / "butterfly_sample_sizes.json"
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nAll sample size results saved to {output_dir / 'butterfly_sample_sizes.json'}")


if __name__ == "__main__":
    main()
