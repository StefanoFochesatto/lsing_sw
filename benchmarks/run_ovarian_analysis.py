#!/usr/bin/env python3
"""
Real-World Biological Graph Recovery:
Ovarian Cancer Transcriptomic Dataset (TCGA / curatedOvarianData, d=156 genes).
Reference: Liaw et al. (AAAI'25), Shutta et al. (2022).

Direct comparison: UMNN Baseline vs Scale-Modulated KAN (SM-KAN).
Evaluates:
- Training wall-clock time
- Generalized precision matrix evaluation time
- Spearman and Pearson correlation between learned gene-gene interactions
- Top-K edge discovery overlap (Jaccard similarity)
- Identification of clinical biomarkers (CTSE, EPCAM) in sparse graph regime
"""

import sys
import time
import json
import argparse
import tracemalloc
import numpy as np
import torch
import networkx as nx
from scipy.stats import spearmanr, pearsonr
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from models.UMNN import MonotonicNN
from models.monotonic_kan import ScaleModulatedKANMap
from trainers.map_trainer import MapTrainer
from trainers.kan_trainer import KANTrainer, KANPrecisionComputer
from evaluators.precision_matrix_computer import PrecisionMatrixComputer


def load_ovarian_data():
    """Loads gene names and standardized train, validation, and test splits."""
    data_dir = REPO_ROOT / "data/ovarian"

    with open(data_dir / "ocDataSmall.txt") as f:
        genes = f.readline().strip().split("\t")

    trn = np.loadtxt(data_dir / "train.txt")
    val = np.loadtxt(data_dir / "validation.txt")
    tst = np.loadtxt(data_dir / "test.txt")

    mean = np.mean(trn, axis=0)
    std = np.std(trn, axis=0) + 1e-8

    trn_std = (trn - mean) / std
    val_std = (val - mean) / std
    tst_std = (tst - mean) / std

    return (
        genes,
        torch.tensor(trn_std, dtype=torch.float32),
        torch.tensor(val_std, dtype=torch.float32),
        torch.tensor(tst_std, dtype=torch.float32),
    )


def compute_centrality_ranks(adj_matrix, genes):
    """
    Computes degree, closeness, betweenness, and eigenvector centrality for graph G,
    and returns ranked list ordered by mean centrality rank.
    """
    d = len(genes)
    G = nx.from_numpy_array(adj_matrix)

    deg_cent = nx.degree_centrality(G)
    close_cent = nx.closeness_centrality(G)
    bet_cent = nx.betweenness_centrality(G)
    try:
        eig_cent = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-4)
    except Exception:
        eig_cent = nx.pagerank(G)

    def _to_ranks(cent_dict):
        sorted_indices = sorted(range(d), key=lambda i: cent_dict[i], reverse=True)
        ranks = np.zeros(d, dtype=float)
        for rank_pos, idx in enumerate(sorted_indices):
            ranks[idx] = rank_pos + 1
        return ranks

    deg_ranks = _to_ranks(deg_cent)
    close_ranks = _to_ranks(close_cent)
    bet_ranks = _to_ranks(bet_cent)
    eig_ranks = _to_ranks(eig_cent)
    mean_ranks = (deg_ranks + close_ranks + bet_ranks + eig_ranks) / 4.0

    sorted_gene_indices = sorted(range(d), key=lambda i: mean_ranks[i])
    gene_rankings = []
    for pos, idx in enumerate(sorted_gene_indices):
        gene_rankings.append({
            "gene": genes[idx],
            "index": int(idx),
            "overall_rank": pos + 1,
            "mean_centrality_rank": float(mean_ranks[idx]),
            "deg_rank": float(deg_ranks[idx]),
            "close_rank": float(close_ranks[idx]),
            "bet_rank": float(bet_ranks[idx]),
            "eig_rank": float(eig_ranks[idx]),
            "degree": int(G.degree[idx]),
        })

    return gene_rankings, G


def main():
    parser = argparse.ArgumentParser(
        description="TCGA Ovarian Cancer Gene Regulatory Discovery: UMNN vs SM-KAN"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--quick", action="store_true", help="Quick mode on subset of genes with fewer epochs")
    parser.add_argument("--max-genes", type=int, default=156, help="Maximum number of genes to train")
    parser.add_argument("--max-epochs", type=int, default=None, help="Max training epochs per model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-umnn", action="store_true", help="Skip UMNN training if already computed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dev_str = args.device
    if dev_str == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        dev_str = "cpu"

    output_dir = REPO_ROOT / "results/kan_benchmarks" / ("quick" if args.quick else "")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"TCGA OVARIAN CANCER GENE EXPRESSION BENCHMARK [Device: {dev_str.upper()}]")
    print("=" * 80)

    all_genes, trn_full, val_full, tst_full = load_ovarian_data()

    if args.quick:
        # Quick demo on 12 features (including CTSE and EPCAM)
        target_genes = ["CTSE", "EPCAM"]
        selected_indices = [all_genes.index(g) for g in target_genes]
        # Fill up to 12 with other genes
        for i in range(len(all_genes)):
            if i not in selected_indices and len(selected_indices) < 12:
                selected_indices.append(i)
        selected_indices = sorted(selected_indices)
        genes = [all_genes[i] for i in selected_indices]
        trn = trn_full[:, selected_indices]
        val = val_full[:, selected_indices]
        tst = tst_full[:, selected_indices]
        d = len(genes)
        regs = [0.05]
        max_epochs = args.max_epochs or 5
        print(f"QUICK MODE: Subsetting to d={d} genes (including CTSE, EPCAM), max_epochs={max_epochs}")
    else:
        d = min(args.max_genes, len(all_genes))
        genes = all_genes[:d]
        trn = trn_full[:, :d]
        val = val_full[:, :d]
        tst = tst_full[:, :d]
        regs = [0.05, 0.01]
        max_epochs = args.max_epochs or 20

    print(f"Features: {d} genes | Split: Train={trn.shape[0]}, Val={val.shape[0]}, Test={tst.shape[0]}")
    print(f"Regularization candidates: {regs} | Max Epochs: {max_epochs}")

    results = {
        "dataset": "TCGA curatedOvarianData",
        "num_genes": d,
        "sample_counts": {"train": trn.shape[0], "val": val.shape[0], "test": tst.shape[0]},
        "device": dev_str,
    }

    # -------------------------------------------------------------------------
    # 1. UMNN Baseline Training & Precision Evaluation
    # -------------------------------------------------------------------------
    umnn_prec_path = output_dir / "ovarian_umnn_precision_matrix.txt"
    if not args.skip_umnn:
        print(f"\n[1/2] Training UMNN models for all {d} genes...")
        tracemalloc.start()
        t0 = time.perf_counter()
        base_umnn = [MonotonicNN(d, [16, 32], 50, dev=dev_str) for _ in range(d)]
        trainer_umnn = MapTrainer(base_umnn, d, [16, 32], 50, 0.01, max_epochs, dev=dev_str)
        umnn_models = []
        for k in range(d):
            others = [j for j in range(d) if j != k]
            best_m, _, _ = trainer_umnn.train_model(trn, k, others, regs, val)
            umnn_models.append(best_m)
            if (k + 1) % 25 == 0 or (k + 1) == d:
                print(f"   UMNN: Trained {k + 1}/{d} genes ({time.perf_counter() - t0:.1f}s elapsed)...")
        umnn_train_time = time.perf_counter() - t0
        _, peak_u = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"Evaluating UMNN precision matrix via autograd...")
        t0 = time.perf_counter()
        ev_u = PrecisionMatrixComputer(umnn_models, tst, d, dev=dev_str)
        umnn_raw = ev_u.compute()
        umnn_sym = ev_u.make_symmetric(umnn_raw)
        umnn_eval_time = time.perf_counter() - t0
        np.savetxt(umnn_prec_path, umnn_sym, fmt="%.6f", delimiter=",")
    else:
        # A quick-run UMNN file (12x12) must never be compared to a full 156-gene KAN run.
        umnn_sym = None
        if umnn_prec_path.exists():
            candidate = np.loadtxt(umnn_prec_path, delimiter=",")
            if candidate.shape == (d, d):
                umnn_sym = candidate
        if umnn_sym is None:
            print(f"\n[1/2] No {d}x{d} UMNN matrix available; skipping paired comparison.")
        else:
            print(f"\n[1/2] Loading matching UMNN matrix from {umnn_prec_path}...")
        umnn_train_time = None
        umnn_eval_time = None
        peak_u = 0

    results["umnn"] = {
        "train_time_sec": umnn_train_time,
        "eval_time_sec": umnn_eval_time,
        "peak_mem_mb": peak_u / (1024 * 1024),
        "precision_matrix_path": str(umnn_prec_path) if umnn_sym is not None else None,
    }

    # -------------------------------------------------------------------------
    # 2. Scale-Modulated Monotonic KAN Training & Precision Evaluation
    # -------------------------------------------------------------------------
    print(f"\n[2/2] Training ScaleModulatedKANMap models for all {d} genes...")
    tracemalloc.start()
    t0 = time.perf_counter()
    base_kan = [
        ScaleModulatedKANMap(d, target_index=k, hidden_dim=4, grid_size=6, grid_range=(-3.5, 3.5), dev=dev_str)
        for k in range(d)
    ]
    trainer_kan = KANTrainer(base_kan, d, hidden_dim=4, grid_size=6, max_epochs=max_epochs, dev=dev_str)
    kan_models = []
    for k in range(d):
        others = [j for j in range(d) if j != k]
        best_m, _, _, _ = trainer_kan.train_model(trn, k, others, regs, val)
        kan_models.append(best_m)
        if (k + 1) % 25 == 0 or (k + 1) == d:
            print(f"   SM-KAN: Trained {k + 1}/{d} genes ({time.perf_counter() - t0:.1f}s elapsed)...")
    kan_train_time = time.perf_counter() - t0
    _, peak_k = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Evaluating SM-KAN precision matrix via exact closed-form derivatives...")
    t0 = time.perf_counter()
    comp_k = KANPrecisionComputer(kan_models, tst.to(dev_str), d)
    kan_raw = comp_k.compute(split=True)
    kan_sym = comp_k.make_symmetric(kan_raw)
    kan_eval_time = time.perf_counter() - t0

    kan_prec_path = output_dir / "ovarian_precision_matrix.txt"
    np.savetxt(kan_prec_path, kan_sym, fmt="%.6f", delimiter=",")

    results["kan"] = {
        "train_time_sec": kan_train_time,
        "eval_time_sec": kan_eval_time,
        "peak_mem_mb": peak_k / (1024 * 1024),
        "precision_matrix_path": str(kan_prec_path),
    }

    # -------------------------------------------------------------------------
    # 3. Direct Statistical & Biological Comparison
    # -------------------------------------------------------------------------
    triu_idx = np.triu_indices(d, k=1)
    kan_offdiag = np.abs(kan_sym[triu_idx])
    sp_corr = pr_corr = None
    top_k_overlap = {}
    if umnn_sym is not None:
        umnn_offdiag = np.abs(umnn_sym[triu_idx])
        sp_corr, _ = spearmanr(umnn_offdiag, kan_offdiag)
        pr_corr, _ = pearsonr(umnn_offdiag, kan_offdiag)
        candidate_k = [k_val for k_val in [10, 25, 50, 100, 200] if k_val < len(umnn_offdiag)]
        for K in candidate_k:
            top_u = set(np.argsort(umnn_offdiag)[-K:])
            top_k = set(np.argsort(kan_offdiag)[-K:])
            intersection = len(top_u & top_k)
            union = len(top_u | top_k)
            top_k_overlap[str(K)] = {
                "overlap_count": intersection,
                "jaccard_similarity": float(intersection / union) if union else 0.0,
            }

    # Graph Sparsity vs Threshold for KAN
    total_pairs = len(kan_offdiag)
    sparsity_report = {}
    for tau in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
        n_edges = int(np.sum(kan_offdiag > tau))
        density = float(n_edges / total_pairs * 100.0)
        sparsity_report[str(tau)] = {"edges_count": n_edges, "density_pct": density}

    # Centrality analysis at tau = 0.10 (sparse regime with active edges)
    tau_chosen = 0.10
    adj_01 = (np.abs(kan_sym) > tau_chosen).astype(int)
    np.fill_diagonal(adj_01, 0)
    rankings_kan, G_kan = compute_centrality_ranks(adj_01, genes)

    # Check CTSE and EPCAM in ranking
    ctse_rank = next((item for item in rankings_kan if item["gene"] == "CTSE"), None)
    epcam_rank = next((item for item in rankings_kan if item["gene"] == "EPCAM"), None)

    comparison = {
        "train_speedup_ratio": (umnn_train_time / kan_train_time) if (umnn_train_time and kan_train_time) else None,
        "eval_speedup_ratio": (umnn_eval_time / kan_eval_time) if (umnn_eval_time and kan_eval_time) else None,
        "spearman_correlation": float(sp_corr) if sp_corr is not None else None,
        "pearson_correlation": float(pr_corr) if pr_corr is not None else None,
        "top_k_overlap": top_k_overlap,
        "kan_sparsity_across_tau": sparsity_report,
        "tau_chosen": tau_chosen,
        "ctse_ranking": ctse_rank,
        "epcam_ranking": epcam_rank,
        "top_10_hub_genes_kan": rankings_kan[:10],
    }
    results["comparison"] = comparison

    out_json = output_dir / "ovarian_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("OVARIAN BENCHMARK SUMMARY (UMNN vs SM-KAN)")
    print("=" * 70)
    if umnn_train_time:
        print(f"Training Time ({d} Genes): UMNN = {umnn_train_time:.1f}s | SM-KAN = {kan_train_time:.1f}s (Speedup: {umnn_train_time/kan_train_time:.2f}x)")
        print(f"Precision Matrix Extraction: UMNN = {umnn_eval_time:.2f}s | SM-KAN = {kan_eval_time:.3f}s (Speedup: {umnn_eval_time/kan_eval_time:.1f}x)")
    if sp_corr is not None:
        print(f"Precision Entry Rank Correlation (Spearman rho): {sp_corr:.3f} (Pearson r = {pr_corr:.3f})")
        for K, info in top_k_overlap.items():
            print(f"  Top-{K:<3} Edge Discovery Overlap: {info['overlap_count']}/{K} shared edges (Jaccard = {info['jaccard_similarity']:.2f})")
    else:
        print("No matched UMNN matrix: cross-method rank correlations and speedups omitted.")
    print(f"\nAll results saved to {out_json}")


if __name__ == "__main__":
    main()
