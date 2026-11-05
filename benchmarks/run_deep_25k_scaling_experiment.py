import argparse
import io
import json
import os
import sys
import time
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.sm_kan_density import SMKANDensityEstimator
from models.real_nvp import RealNVPFlow
from models.UMNN.UMNNMAFFlow import UMNNMAFFlow
from src.datasets.synthetic_2d import get_2d_dataloaders


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_computational_scaling():
    """
    Empirically benchmarks forward and backward pass latency as depth K increases
    to verify that composite KAN normalizing flows scale strictly O(K) without
    nested derivative explosion.
    """
    print("\n-------------------------------------------------------------")
    print("  1. EMPIRICAL SCALING AUDIT: LATENCY VS. DEPTH K")
    print("-------------------------------------------------------------")
    x = torch.randn(256, 2)
    depths = [1, 2, 4, 8, 14]
    latencies = []

    for K in depths:
        model = SMKANDensityEstimator(num_features=2, num_steps=K, hidden_dim=32, grid_size=8)
        p_cnt = count_parameters(model)

        # Warmup
        _ = model.log_prob(x)

        # Forward latency
        t0 = time.perf_counter()
        n_eval = 25
        for _ in range(n_eval):
            lp = model.log_prob(x)
        t_fwd = (time.perf_counter() - t0) / n_eval

        # Backward latency
        t0 = time.perf_counter()
        for _ in range(n_eval):
            loss = -model.log_prob(x).mean()
            loss.backward()
        t_bwd = (time.perf_counter() - t0) / n_eval

        per_step_fwd = (t_fwd * 1000.0) / K
        latencies.append({
            "K": K,
            "params": p_cnt,
            "fwd_ms": t_fwd * 1000.0,
            "bwd_ms": t_bwd * 1000.0,
            "per_step_ms": per_step_fwd,
        })
        print(f"  Depth K={K:2d} ({p_cnt:6,d} params) -> Fwd: {t_fwd*1000:6.2f} ms | Bwd: {t_bwd*1000:6.2f} ms | Per-Step: {per_step_fwd:5.2f} ms")

    return latencies


def train_eval_model(
    model: nn.Module,
    train_loader,
    val_loader,
    test_tensor: torch.Tensor,
    epochs: int = 12,
    lr: float = 4e-3,
    device: str = "cpu",
    is_umnn: bool = False,
) -> Tuple[float, float, float, float]:
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)

    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        for (batch,) in train_loader:
            batch = batch.to(device)
            if is_umnn:
                ll, _ = model.compute_ll(batch)
                loss = -ll.mean()
            else:
                loss = -model.log_prob(batch).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    train_time = time.perf_counter() - t0

    model.eval()
    test_tensor = test_tensor.to(device)
    with torch.no_grad():
        if is_umnn:
            ll_trn, _ = model.compute_ll(train_loader.dataset.tensors[0].to(device))
            train_ll = ll_trn.mean().item()
            ll_tst, _ = model.compute_ll(test_tensor)
            test_ll = ll_tst.mean().item()
        else:
            chunk = 1000
            lps_tr = [model.log_prob(train_loader.dataset.tensors[0][i : i + chunk].to(device)) for i in range(0, len(train_loader.dataset), chunk)]
            train_ll = torch.cat(lps_tr, dim=0).mean().item()
            lps_te = [model.log_prob(test_tensor[i : i + chunk]) for i in range(0, test_tensor.shape[0], chunk)]
            test_ll = torch.cat(lps_te, dim=0).mean().item()

    # Sampling speed
    t0 = time.perf_counter()
    with torch.no_grad():
        if is_umnn:
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                _ = model.invert(torch.randn(500, 2, device=device), iter=15)
            finally:
                sys.stdout = old_stdout
            sample_time = (time.perf_counter() - t0) * 2.0
        else:
            _ = model.sample(1000, device=device)
            sample_time = time.perf_counter() - t0

    return float(train_ll), float(test_ll), float(train_time), float(sample_time)


def evaluate_grid(model: nn.Module, res: int = 100, lim: float = 3.6, is_umnn: bool = False, device: str = "cpu"):
    xs = torch.linspace(-lim, lim, res)
    ys = torch.linspace(-lim, lim, res)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    pts = torch.stack([X.flatten(), Y.flatten()], dim=-1).to(device)

    model.eval()
    with torch.no_grad():
        chunk = 2000
        lps = []
        for i in range(0, pts.shape[0], chunk):
            p = pts[i : i + chunk]
            if is_umnn:
                lp, _ = model.compute_ll(p)
            else:
                lp = model.log_prob(p)
            lps.append(lp.cpu())
        full_lp = torch.cat(lps, dim=0)
        prob = torch.exp(full_lp).reshape(res, res).numpy()
    return X.numpy(), Y.numpy(), prob


def generate_samples(model: nn.Module, n_samples: int = 1000, is_umnn: bool = False, device: str = "cpu"):
    model.eval()
    with torch.no_grad():
        if is_umnn:
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                samples = model.invert(torch.randn(n_samples, 2, device=device), iter=15).cpu().numpy()
            finally:
                sys.stdout = old_stdout
        else:
            samples = model.sample(n_samples, device=device).cpu().numpy()
    return samples


def create_comparison_figure(
    dataset_name: str,
    target_data: torch.Tensor,
    models: Dict[str, Tuple[nn.Module, bool, float, float, int]],
    save_path: str,
):
    fig, axes = plt.subplots(2, 4, figsize=(20, 9.5))
    keys = list(models.keys())
    colors = ["#2563eb", "#16a34a", "#dc2626"]

    # Col 0: Ground Truth
    axes[0, 0].scatter(target_data[:, 0], target_data[:, 1], s=3, alpha=0.35, color="#0f172a")
    axes[0, 0].set_title(f"Target Distribution: {dataset_name.upper()}\n(Ground Truth Empirical Density)", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlim(-3.6, 3.6); axes[0, 0].set_ylim(-3.6, 3.6); axes[0, 0].set_aspect("equal")

    axes[1, 0].scatter(target_data[:1000, 0], target_data[:1000, 1], s=5, alpha=0.5, color="#0f172a")
    axes[1, 0].set_title("Target Samples (N=1,000)\nReference Ground Truth", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlim(-3.6, 3.6); axes[1, 0].set_ylim(-3.6, 3.6); axes[1, 0].set_aspect("equal")

    # Cols 1..3: Models
    for col, k in enumerate(keys, start=1):
        model, is_umnn, trn_ll, tst_ll, p_cnt = models[k]

        # Density Contours
        ax_top = axes[0, col]
        X, Y, prob = evaluate_grid(model, res=90, lim=3.6, is_umnn=is_umnn)
        cf = ax_top.contourf(X, Y, prob, levels=35, cmap="magma")
        fig.colorbar(cf, ax=ax_top, fraction=0.046, pad=0.04)
        ax_top.set_title(f"{k}\nTest LL: {tst_ll:+.3f} nats | {p_cnt:,} params", fontsize=11, fontweight="bold")
        ax_top.set_xlim(-3.6, 3.6); ax_top.set_ylim(-3.6, 3.6); ax_top.set_aspect("equal")

        # Generated Samples
        ax_bot = axes[1, col]
        samps = generate_samples(model, n_samples=1000, is_umnn=is_umnn)
        ax_bot.scatter(samps[:, 0], samps[:, 1], s=5, alpha=0.5, color=colors[col - 1])
        ax_bot.set_title(f"{k}\nGenerated Samples (N=1,000)", fontsize=10)
        ax_bot.set_xlim(-3.6, 3.6); ax_bot.set_ylim(-3.6, 3.6); ax_bot.set_aspect("equal")

    plt.suptitle(
        f"Deep KAN Scaling: Scaling Flow Depth to 25,000+ Parameters (K=14 Steps, G=8 Knots) vs. RealNVP (26.5k Params)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Successfully saved figure: {save_path}")


def run_experiment(output_dir: str = "results/kan_benchmarks"):
    os.makedirs(output_dir, exist_ok=True)
    device = "cpu"

    # Step 1: Computational Scaling Audit
    latencies = benchmark_computational_scaling()

    # Step 2: Density Estimation on Two Moons
    print("\n-------------------------------------------------------------")
    print("  2. TRAINING DEEP 25,000+ PARAMETER KAN ON TWO MOONS")
    print("-------------------------------------------------------------")
    tr, va, te, test_t, full_data = get_2d_dataloaders("moons", n_train=4000, n_val=1000, n_test=1000, batch_size=256, seed=42)

    # 1. Deep 25k KAN (K=14 steps, Q=32, G=8 -> 25,004 params)
    print("  --> Training Deep SM-KAN (K=14 steps, G=8 knots, 25,004 params)...")
    m_deep_25k = SMKANDensityEstimator(num_features=2, num_steps=14, hidden_dim=32, grid_size=8)
    trn_ll_k14, tst_ll_k14, t_train_k14, t_samp_k14 = train_eval_model(m_deep_25k, tr, va, test_t, epochs=12, lr=4e-3, device=device)
    print(f"      Train LL: {trn_ll_k14:+.3f} | Test LL: {tst_ll_k14:+.3f} | Train Time: {t_train_k14:.2f}s | 1k Samp Time: {t_samp_k14*1000:.1f}ms")

    # 2. Compact KAN (K=1 step, Q=6, G=8 -> 356 params)
    print("  --> Training Compact SM-KAN (K=1 step, 356 params)...")
    m_compact = SMKANDensityEstimator(num_features=2, num_steps=1, hidden_dim=6, grid_size=8)
    trn_ll_c, tst_ll_c, t_train_c, t_samp_c = train_eval_model(m_compact, tr, va, test_t, epochs=12, lr=8e-3, device=device)
    print(f"      Train LL: {trn_ll_c:+.3f} | Test LL: {tst_ll_c:+.3f} | Train Time: {t_train_c:.2f}s | 1k Samp Time: {t_samp_c*1000:.1f}ms")

    # 3. RealNVP (6 layers -> 26,508 params)
    print("  --> Training RealNVP Baseline (26,508 params)...")
    m_nvp = RealNVPFlow(num_features=2, num_layers=6, hidden_dim=64)
    trn_ll_nvp, tst_ll_nvp, t_train_nvp, t_samp_nvp = train_eval_model(m_nvp, tr, va, test_t, epochs=12, lr=3e-3, device=device)
    print(f"      Train LL: {trn_ll_nvp:+.3f} | Test LL: {tst_ll_nvp:+.3f} | Train Time: {t_train_nvp:.2f}s | 1k Samp Time: {t_samp_nvp*1000:.1f}ms")

    models_dict = {
        "Compact SM-KAN (K=1)": (m_compact, False, trn_ll_c, tst_ll_c, count_parameters(m_compact)),
        "Deep SM-KAN (K=14, 25k P)": (m_deep_25k, False, trn_ll_k14, tst_ll_k14, count_parameters(m_deep_25k)),
        "RealNVP Baseline": (m_nvp, False, trn_ll_nvp, tst_ll_nvp, count_parameters(m_nvp)),
    }

    fig_path = os.path.join(output_dir, "density_deep_25k_moons.png")
    create_comparison_figure("moons", full_data[:2000], models_dict, fig_path)

    results_data = {
        "latencies_vs_depth": latencies,
        "two_moons": {
            "deep_smkan_k14": {
                "params": count_parameters(m_deep_25k),
                "train_ll": trn_ll_k14,
                "test_ll": tst_ll_k14,
                "gap": trn_ll_k14 - tst_ll_k14,
                "train_time_sec": t_train_k14,
                "sample_time_1000": t_samp_k14,
            },
            "compact_smkan_k1": {
                "params": count_parameters(m_compact),
                "train_ll": trn_ll_c,
                "test_ll": tst_ll_c,
                "gap": trn_ll_c - tst_ll_c,
                "train_time_sec": t_train_c,
                "sample_time_1000": t_samp_c,
            },
            "real_nvp": {
                "params": count_parameters(m_nvp),
                "train_ll": trn_ll_nvp,
                "test_ll": tst_ll_nvp,
                "gap": trn_ll_nvp - tst_ll_nvp,
                "train_time_sec": t_train_nvp,
                "sample_time_1000": t_samp_nvp,
            },
        },
    }

    out_json = os.path.join(output_dir, "deep_25k_experiment_results.json")
    with open(out_json, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nAll deep 25k experiment results saved to: {out_json}")


if __name__ == "__main__":
    run_experiment()
