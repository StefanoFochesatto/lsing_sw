#!/usr/bin/env python3
"""
Real-World Biological Graph Recovery:
Ovarian Cancer Transcriptomic Dataset (TCGA / curatedOvarianData, d=156 genes).
Reference: Liaw et al. (AAAI'25), Shutta et al. (2022).

Evaluates whether Monotonic KAN transport maps discover known biological biomarkers:
- CTSE (Cathepsin E): Mucinous ovarian tumor marker (Marquez et al., 2005)
- EPCAM: Epithelial cell adhesion molecule (Spizzo et al., 2006)
using graph centrality metrics across the 156-gene network.
"""

import sys
import time
import json
import tracemalloc
import numpy as np
import torch
import networkx as nx
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from models.monotonic_kan import ScaleModulatedKANMap
from trainers.kan_trainer import KANTrainer, KANPrecisionComputer


def load_ovarian_data():
    """Loads gene names and standardized train, validation, and test splits."""
    data_dir = REPO_ROOT / "data/ovarian"

    with open(data_dir / "ocDataSmall.txt") as f:
        genes = f.readline().strip().split("\t")

    trn = np.loadtxt(data_dir / "train.txt")
    val = np.loadtxt(data_dir / "validation.txt")
    tst = np.loadtxt(data_dir / "test.txt")

    # Standardize features using training set statistics
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

    # 1. Degree centrality
    deg_cent = nx.degree_centrality(G)
    # 2. Closeness centrality
    close_cent = nx.closeness_centrality(G)
    # 3. Betweenness centrality
    bet_cent = nx.betweenness_centrality(G)
    # 4. Eigenvector / Hub score centrality
    try:
        eig_cent = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-4)
    except Exception:
        eig_cent = nx.pagerank(G)

    # Convert to ranks (1 = highest centrality)
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
    torch.manual_seed(42)
    np.random.seed(42)

    output_dir = REPO_ROOT / "results/kan_benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("REAL-WORLD OVARIAN CANCER GENE EXPRESSION BENCHMARK (d=156)")
    print("=" * 80)

    genes, trn, val, tst = load_ovarian_data()
    d = len(genes)
    print(f"Loaded {d} genes from TCGA curatedOvarianData.")
    print(f"Split sizes: Train={trn.shape[0]}, Validation={val.shape[0]}, Test={tst.shape[0]}")
    print(f"Target biomarkers: CTSE (index {genes.index('CTSE')}), EPCAM (index {genes.index('EPCAM')})")

    regs = [0.05, 0.01]
    max_epochs = 20

    print(f"\nTraining ScaleModulatedKANMap across all {d} genes...")
    tracemalloc.start()
    t0 = time.perf_counter()

    base_map = [
        ScaleModulatedKANMap(d, target_index=k, hidden_dim=4, grid_size=6, grid_range=(-3.5, 3.5))
        for k in range(d)
    ]
    trainer = KANTrainer(base_map, d, hidden_dim=4, grid_size=6, max_epochs=max_epochs)

    trained_models = []
    for k in range(d):
        others = [j for j in range(d) if j != k]
        best_m, opt_reg, _, _ = trainer.train_model(trn, k, others, regs, val)
        trained_models.append(best_m)
        if (k + 1) % 25 == 0 or (k + 1) == d:
            print(f"   Trained {k + 1}/{d} genes (elapsed: {time.perf_counter() - t0:.1f} s)...")

    train_time = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_mb = peak / (1024 * 1024)

    print(f"\nAll {d} gene transport maps trained in {train_time:.1f} s ({train_time/d:.3f} s/gene)!")
    print(f"Peak memory consumption: {peak_mem_mb:.2f} MB.")

    print("\nEvaluating closed-form generalized precision matrix on 117 evaluation samples...")
    t0 = time.perf_counter()
    comp = KANPrecisionComputer(trained_models, tst, d)
    raw_prec = comp.compute()
    sym_prec = comp.make_symmetric(raw_prec)
    eval_time = time.perf_counter() - t0
    print(f"Precision matrix ({d}x{d} = {d*d} entries) computed in {eval_time:.3f} s!")

    # Save precision matrix
    prec_path = output_dir / "ovarian_precision_matrix.txt"
    np.savetxt(prec_path, sym_prec, fmt="%.6f", delimiter=",")
    print(f"Precision matrix saved to {prec_path}")

    # Sparsity analysis across thresholds
    triu_idx = np.triu_indices(d, k=1)
    off_diag_vals = np.abs(sym_prec[triu_idx])

    print("\n--- Graph Sparsity vs. Threshold tau ---")
    print(f"{'tau':<8} | {'Edges Count':<12} | {'Density (%)':<12} | {'Entries < tau (%)':<18}")
    print("-" * 55)
    total_pairs = len(off_diag_vals)
    sparsity_report = {}
    for tau in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        n_edges = int(np.sum(off_diag_vals > tau))
        pct_less = float(np.mean(off_diag_vals < tau) * 100.0)
        density = float(n_edges / total_pairs * 100.0)
        sparsity_report[str(tau)] = {
            "edges_count": n_edges,
            "density_pct": density,
            "entries_below_pct": pct_less,
        }
        print(f"{tau:<8.2f} | {n_edges:<12} | {density:<12.1f} | {pct_less:<18.1f}")

    # Centrality analysis at tau = 0.20 (Paper's recommended threshold)
    tau_chosen = 0.20
    adj_02 = (np.abs(sym_prec) > tau_chosen).astype(int)
    np.fill_diagonal(adj_02, 0)

    rankings, G = compute_centrality_ranks(adj_02, genes)

    print(f"\n--- Graph Centrality Analysis at tau = {tau_chosen} (Total Edges: {G.number_of_edges()}) ---")
    print(f"{'Rank':<6} | {'Gene':<12} | {'Mean Rank':<12} | {'Degree':<8} | {'Deg Rank':<10} | {'Close Rank':<12} | {'Bet Rank':<10} | {'Eig Rank':<10}")
    print("-" * 90)
    for r in rankings[:15]:
        print(f"{r['overall_rank']:<6} | {r['gene']:<12} | {r['mean_centrality_rank']:<12.1f} | {r['degree']:<8} | {r['deg_rank']:<10.1f} | {r['close_rank']:<12.1f} | {r['bet_rank']:<10.1f} | {r['eig_rank']:<10.1f}")

    # Specific check on CTSE and EPCAM
    ctse_info = next(item for item in rankings if item["gene"] == "CTSE")
    epcam_info = next(item for item in rankings if item["gene"] == "EPCAM")

    print("\n--- Key Biological Biomarkers ---")
    print(f"CTSE (Cathepsin E, Mucinous Marker):")
    print(f"   Overall Rank: {ctse_info['overall_rank']}/156 | Mean Centrality Rank: {ctse_info['mean_centrality_rank']:.1f} | Degree: {ctse_info['degree']}")
    print(f"EPCAM (Epithelial Adhesion Marker):")
    print(f"   Overall Rank: {epcam_info['overall_rank']}/156 | Mean Centrality Rank: {epcam_info['mean_centrality_rank']:.1f} | Degree: {epcam_info['degree']}")

    results = {
        "dataset": "TCGA curatedOvarianData",
        "num_genes": d,
        "sample_counts": {"train": trn.shape[0], "val": val.shape[0], "test": tst.shape[0]},
        "training_time_sec": train_time,
        "eval_time_sec": eval_time,
        "peak_mem_mb": peak_mem_mb,
        "sparsity_across_tau": sparsity_report,
        "tau_chosen": tau_chosen,
        "ctse_ranking": ctse_info,
        "epcam_ranking": epcam_info,
        "top_15_hub_genes": rankings[:15],
    }

    out_json = output_dir / "ovarian_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {out_json}")


if __name__ == "__main__":
    main()
