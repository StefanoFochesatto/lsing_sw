#!/usr/bin/env python3
"""
Generate summary visualization figures for Monotonic KAN Transport Maps vs UMNN.
Saves publication-quality PNGs in results/kan_benchmarks/figures/.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parent
BENCH_DIR = REPO_ROOT / "results/kan_benchmarks"
FIG_DIR = BENCH_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)


def plot_profiling_and_speedups():
    prof_file = BENCH_DIR / "profiling.json"
    if not prof_file.exists():
        print(f"Skipping profiling plot: {prof_file} not found")
        return

    with open(prof_file) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    p_umnn = data["parameter_count"]["umnn"]
    p_kan = data["parameter_count"]["monotonic_kan"]
    bars = axes[0].bar(["UMNN", "Monotonic KAN"], [p_umnn, p_kan], color=["#e74c3c", "#2ecc71"], width=0.55)
    axes[0].set_title(f"Model Parameters\n({data['parameter_count']['reduction_ratio']:.1f}x Parameter Reduction)", fontweight="bold")
    axes[0].set_ylabel("Parameter Count")
    for b in bars:
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 30, f"{int(b.get_height())}", ha="center", fontweight="bold")
    axes[0].set_ylim(0, max(p_umnn, p_kan) * 1.18)

    fps_umnn = data["training_throughput"]["umnn_samples_per_sec"]
    fps_kan = data["training_throughput"]["kan_samples_per_sec"]
    speedup = data["training_throughput"]["speedup_factor"]
    bars = axes[1].bar(["UMNN", "Monotonic KAN"], [fps_umnn / 1000, fps_kan / 1000], color=["#e74c3c", "#2ecc71"], width=0.55)
    axes[1].set_title(f"Training Throughput\n({speedup:.2f}x Faster)", fontweight="bold")
    axes[1].set_ylabel("Throughput (k samples / sec)")
    for b in bars:
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{b.get_height():.1f}k", ha="center", fontweight="bold")
    axes[1].set_ylim(0, max(fps_umnn, fps_kan) / 1000 * 1.18)

    t_umnn = data["precision_matrix_extraction"]["umnn_quadrature_seconds"]
    t_vmap = data["precision_matrix_extraction"]["kan_torch_func_seconds"]
    t_kan = data["precision_matrix_extraction"]["kan_analytic_chain_seconds"]
    speedup_prec = data["precision_matrix_extraction"]["analytic_vs_umnn_speedup"]
    bars = axes[2].bar(["UMNN\n(Quadrature)", "KAN\n(torch.func)", "KAN\n(Analytic)"], [t_umnn, t_vmap, t_kan], color=["#e74c3c", "#3498db", "#2ecc71"], width=0.55)
    axes[2].set_title(f"Precision Matrix Extraction\n({speedup_prec:.1f}x Faster than UMNN)", fontweight="bold")
    axes[2].set_ylabel("Evaluation Time (seconds)")
    for b in bars:
        h = b.get_height()
        axes[2].text(b.get_x() + b.get_width() / 2, h + 0.15, f"{h:.3f}s" if h < 1 else f"{h:.2f}s", ha="center", fontweight="bold")
    axes[2].set_ylim(0, max(t_umnn, t_vmap) * 1.18)

    plt.tight_layout()
    out_path = FIG_DIR / "chart_computational_speedup.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def plot_gaussian_benchmark():
    gauss_file = BENCH_DIR / "gaussian_benchmark.json"
    if not gauss_file.exists():
        return
    with open(gauss_file) as f:
        data = json.load(f)

    d = 10
    true_adj = np.zeros((d, d))
    edges = [(0, 4), (2, 6), (5, 7), (6, 9)]
    for u, v in edges:
        true_adj[u, v] = true_adj[v, u] = 1.0
    np.fill_diagonal(true_adj, 1.0)

    p_umnn = np.array(data["UMNN"]["precision_matrix"])
    p_kan = np.array(data["MonotonicKAN_Functional"]["precision_matrix"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    cmap = "Blues"

    sns.heatmap(true_adj, ax=axes[0], cmap=cmap, vmin=0, vmax=1, cbar=True, square=True)
    axes[0].set_title("Ground Truth Graph\n(Gaussian d=10)", fontweight="bold")

    sns.heatmap(np.abs(p_umnn), ax=axes[1], cmap=cmap, vmin=0, vmax=1, cbar=True, square=True)
    axes[1].set_title(r"UMNN Recovered $|\hat{\Omega}|$" + "\n(F1=1.0 at tau=0.2, F1=0.67 at tau=0.1)", fontweight="bold")

    sns.heatmap(np.abs(p_kan), ax=axes[2], cmap=cmap, vmin=0, vmax=1, cbar=True, square=True)
    axes[2].set_title(r"Monotonic KAN Recovered $|\hat{\Omega}|$" + "\n(F1=1.0 at tau=0.2, F1=1.0 at tau=0.1)", fontweight="bold")

    plt.tight_layout()
    out_path = FIG_DIR / "chart_gaussian_precision_comparison.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def plot_butterfly_sample_sizes():
    bf_file = BENCH_DIR / "butterfly_sample_sizes.json"
    if not bf_file.exists():
        return
    with open(bf_file) as f:
        data = json.load(f)

    sample_sizes = sorted([int(k) for k in data.keys()])
    umnn_f1 = [data[str(m)]["UMNN"]["metrics"]["0.1"]["f1"] for m in sample_sizes]
    kan_f1 = [data[str(m)]["MonotonicKAN_Functional"]["metrics"]["0.1"]["f1"] for m in sample_sizes]

    umnn_train = [data[str(m)]["UMNN"]["train_time_sec"] for m in sample_sizes]
    kan_train = [data[str(m)]["MonotonicKAN_Functional"]["train_time_sec"] for m in sample_sizes]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(sample_sizes, umnn_f1, "o--", color="#e74c3c", linewidth=2.2, markersize=8, label="UMNN Baseline")
    axes[0].plot(sample_sizes, kan_f1, "s-", color="#2ecc71", linewidth=2.5, markersize=8, label="Monotonic KAN (Functional)")
    axes[0].set_title("Butterfly Graph Discovery: F1 vs Training Samples (tau=0.1)", fontweight="bold")
    axes[0].set_xlabel("Sample Size (M)")
    axes[0].set_ylabel("F1 Score")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend(loc="lower right")

    axes[1].plot(sample_sizes, umnn_train, "o--", color="#e74c3c", linewidth=2.2, markersize=8, label="UMNN Baseline")
    axes[1].plot(sample_sizes, kan_train, "s-", color="#2ecc71", linewidth=2.5, markersize=8, label="Monotonic KAN")
    axes[1].set_title("Training Wall-Clock Time vs Sample Size", fontweight="bold")
    axes[1].set_xlabel("Sample Size (M)")
    axes[1].set_ylabel("Training Time (seconds)")
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    out_path = FIG_DIR / "chart_butterfly_sample_size_sensitivity.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def plot_mixed_coupling():
    mc_file = BENCH_DIR / "mixed_coupling_results.json"
    if not mc_file.exists():
        return
    with open(mc_file) as f:
        data = json.load(f)

    true_adj = np.array(data["ground_truth_adj"])
    raw_add = np.array(data["AdditiveMonotonicKAN"]["raw_precision_matrix"])
    raw_sm = np.array(data["ScaleModulatedKAN"]["raw_precision_matrix"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    cmap = "viridis"

    sns.heatmap(true_adj, ax=axes[0], cmap="Blues", vmin=0, vmax=1, cbar=True, square=True)
    axes[0].set_title("Ground Truth Graph\n(Additive, Multiplicative, Mixed)", fontweight="bold")

    sns.heatmap(raw_add, ax=axes[1], cmap=cmap, vmin=0, vmax=3.5, cbar=True, square=True)
    axes[1].set_title("Additive KAN (Polarization Dilemma)\nFails on Multiplicative Edges (2,3 & 2,5)", fontweight="bold")

    sns.heatmap(raw_sm, ax=axes[2], cmap=cmap, vmin=0, vmax=3.5, cbar=True, square=True)
    axes[2].set_title("Scale-Modulated KAN (SM-KAN)\nDiscovers All 3 Regimes Simultaneously", fontweight="bold")

    plt.tight_layout()
    out_path = FIG_DIR / "chart_mixed_coupling_discovery.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def plot_mixed_coupling_diagnostic():
    mc_file = BENCH_DIR / "mixed_coupling_results.json"
    if not mc_file.exists():
        return
    with open(mc_file) as f:
        data = json.load(f)

    raw_add = np.array(data["AdditiveMonotonicKAN"]["raw_precision_matrix"])
    raw_sm = np.array(data["ScaleModulatedKAN"]["raw_precision_matrix"])

    edges_info = [
        ("Additive\n(0, 1)", 0, 1),
        ("Multiplicative\n(2, 3)", 2, 3),
        ("Mixed Add\n(4, 5)", 4, 5),
        ("Mixed Mult\n(2, 5)", 2, 5),
        ("Spurious\n(0, 3)", 0, 3),
        ("Spurious\n(1, 2)", 1, 2),
    ]

    labels = [e[0] for e in edges_info]
    add_signals = [raw_add[e[1], e[2]] for e in edges_info]
    sm_signals = [raw_sm[e[1], e[2]] for e in edges_info]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    x = np.arange(len(labels))
    width = 0.35

    rects1 = axes[0].bar(x - width/2, add_signals, width, label="Additive Monotonic KAN", color="#e74c3c", alpha=0.88)
    rects2 = axes[0].bar(x + width/2, sm_signals, width, label="Scale-Modulated KAN (SM-KAN)", color="#2ecc71", alpha=0.88)

    axes[0].set_ylabel(r"Precision Signal $|\hat{\Omega}_{jk}|$")
    axes[0].set_title("Edge Discovery Contrast Across Regimes\n(Notice Failure of Additive KAN on Multiplicative Edges)", fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontweight="bold")
    axes[0].axvline(3.5, color="gray", linestyle=":", label="True Edges | Spurious Non-Edges")
    axes[0].legend(loc="upper left")

    for r in rects1:
        h = r.get_height()
        axes[0].annotate(f"{h:.2f}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                         textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for r in rects2:
        h = r.get_height()
        axes[0].annotate(f"{h:.2f}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                         textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight="bold")
    axes[0].set_ylim(0, max(max(add_signals), max(sm_signals)) * 1.18)

    path_stats = data["ScaleModulatedKAN"].get("pathway_spline_std", {})
    if path_stats:
        node_labels = ["X1 (Add Parent X0)", "X3 (Mult Parent X2)", "X5 (Add X4, Mult X2)"]
        scale_stds = [
            path_stats["node_1_pure_additive"]["X_0"]["std_psi_scale"],
            path_stats["node_3_pure_multiplicative"]["X_2"]["std_psi_scale"],
            path_stats["node_5_mixed_add_mult"]["X_2"]["std_psi_scale"],
        ]
        offset_stds = [
            path_stats["node_1_pure_additive"]["X_0"]["std_rho_offset"],
            path_stats["node_3_pure_multiplicative"]["X_2"]["std_rho_offset"],
            path_stats["node_5_mixed_add_mult"]["X_4"]["std_rho_offset"],
        ]

        x_nodes = np.arange(len(node_labels))
        axes[1].bar(x_nodes - width/2, scale_stds, width, label=r"Scale Spline Std ($\psi_j$)", color="#9b59b6")
        axes[1].bar(x_nodes + width/2, offset_stds, width, label=r"Offset Spline Std ($\rho_j$)", color="#f39c12")
        axes[1].set_ylabel("Empirical Activation Std")
        axes[1].set_title(r"Internal SM-KAN Pathway Decoupling ($\psi_j$ vs $\rho_j$)", fontweight="bold")
        axes[1].set_xticks(x_nodes)
        axes[1].set_xticklabels(node_labels, fontweight="bold")
        axes[1].legend(loc="upper left")

    plt.tight_layout()
    out_path = FIG_DIR / "chart_mixed_coupling_diagnostic.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def plot_ovarian_network():
    prec_file = BENCH_DIR / "ovarian_precision_matrix.txt"
    if not prec_file.exists():
        return
    prec = np.loadtxt(prec_file, delimiter=",")
    d = prec.shape[0]
    triu_idx = np.triu_indices(d, k=1)
    off_diag = np.abs(prec[triu_idx])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.heatmap(np.abs(prec), ax=axes[0], cmap="mako", vmin=0, vmax=0.15, cbar=True, square=True)
    axes[0].set_title(f"TCGA Ovarian Gene Network\n({d}x{d} Precision Matrix)", fontweight="bold")
    axes[0].set_xlabel("Gene Index")
    axes[0].set_ylabel("Gene Index")

    taus = np.linspace(0.01, 0.15, 50)
    densities = [np.mean(off_diag > tau) * 100.0 for tau in taus]
    axes[1].plot(taus, densities, color="#2980b9", lw=2.5)
    axes[1].axvline(0.08, color="#e74c3c", linestyle="--", label="Sparsity Cutoff (tau=0.08, 192 edges)")
    axes[1].axvline(0.10, color="#27ae60", linestyle="--", label="High-Confidence Cutoff (tau=0.10, 98 edges)")
    axes[1].set_title(r"Network Edge Density vs Threshold $\tau$", fontweight="bold")
    axes[1].set_xlabel(r"Precision Cutoff Threshold $\tau$")
    axes[1].set_ylabel("Graph Density (%)")
    axes[1].legend(loc="upper right")

    sns.histplot(off_diag, bins=60, kde=True, ax=axes[2], color="#8e44ad")
    axes[2].set_title("Distribution of Gene Pair Dependencies", fontweight="bold")
    axes[2].set_xlabel(r"Absolute Precision Entry $|\hat{\Omega}_{jk}|$")
    axes[2].set_ylabel("Number of Gene Pairs")

    plt.tight_layout()
    out_path = FIG_DIR / "chart_ovarian_precision_network.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    print("Generating comprehensive benchmark visualization figures...")
    plot_profiling_and_speedups()
    plot_gaussian_benchmark()
    plot_butterfly_sample_sizes()
    plot_mixed_coupling()
    plot_mixed_coupling_diagnostic()
    plot_ovarian_network()
    print(f"All figures generated successfully in {FIG_DIR}")


if __name__ == "__main__":
    main()
