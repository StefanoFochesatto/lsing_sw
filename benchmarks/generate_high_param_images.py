import argparse
import io
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


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int = 15,
    lr: float = 8e-3,
    device: str = "cpu",
    is_umnn: bool = False,
) -> nn.Module:
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)

    best_val_loss = float("inf")
    best_state = None

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

        # Validation
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                if is_umnn:
                    ll, _ = model.compute_ll(batch)
                    val_loss += (-ll.mean().item())
                else:
                    val_loss += (-model.log_prob(batch).mean().item())
                val_count += 1
        val_loss /= max(val_count, 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return model


def evaluate_density_grid(
    model: nn.Module,
    grid_res: int = 100,
    range_lim: float = 3.8,
    device: str = "cpu",
    is_umnn: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = torch.linspace(-range_lim, range_lim, grid_res)
    ys = torch.linspace(-range_lim, range_lim, grid_res)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    pts = torch.stack([X.flatten(), Y.flatten()], dim=-1).to(device)

    model.eval()
    with torch.no_grad():
        chunk_size = 2000
        lps = []
        for i in range(0, pts.shape[0], chunk_size):
            chunk = pts[i : i + chunk_size]
            if is_umnn:
                lp, _ = model.compute_ll(chunk)
            else:
                lp = model.log_prob(chunk)
            lps.append(lp.cpu())
        full_lp = torch.cat(lps, dim=0)
        prob = torch.exp(full_lp).reshape(grid_res, grid_res).numpy()

    return X.numpy(), Y.numpy(), prob


def generate_samples(
    model: nn.Module,
    n_samples: int = 1000,
    device: str = "cpu",
    is_umnn: bool = False,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        if is_umnn:
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                z = torch.randn(n_samples, 2, device=device)
                samples = model.invert(z, iter=15).cpu().numpy()
            finally:
                sys.stdout = old_stdout
        else:
            samples = model.sample(n_samples, device=device).cpu().numpy()
    return samples


def create_high_param_comparison_figure(
    dataset_name: str,
    target_data: torch.Tensor,
    models_dict: Dict[str, Tuple[nn.Module, bool, float, int]],
    save_path: str,
):
    """
    Creates a 2x5 publication-quality comparison figure:
      Row 1: Density Contours
      Row 2: Generated Samples
    Columns:
      1: Ground Truth
      2: SM-KAN Compact (~350 params)
      3: SM-KAN High-Capacity (~24.5k params) [Parameter-Matched to RealNVP]
      4: RealNVP (~26.5k params)
      5: UMNN-MAF (~5.5k params)
    """
    fig, axes = plt.subplots(2, 5, figsize=(24, 9.5))

    keys = list(models_dict.keys())
    colors = ["#2563eb", "#16a34a", "#dc2626", "#9333ea"]

    # -------------------------------------------------------------
    # Col 0: Ground Truth Target Distribution
    # -------------------------------------------------------------
    # Row 1, Col 0: Target scatter preview
    axes[0, 0].scatter(target_data[:, 0], target_data[:, 1], s=3, alpha=0.35, color="#0f172a")
    axes[0, 0].set_title(f"Target Distribution: {dataset_name.upper()}\n(Ground Truth Empirical Density)", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlim(-4, 4)
    axes[0, 0].set_ylim(-4, 4)
    axes[0, 0].set_aspect("equal")

    # Row 2, Col 0: Target samples
    axes[1, 0].scatter(target_data[:1000, 0], target_data[:1000, 1], s=5, alpha=0.5, color="#0f172a")
    axes[1, 0].set_title("Target Samples (N=1,000)\nGround Truth Reference", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlim(-4, 4)
    axes[1, 0].set_ylim(-4, 4)
    axes[1, 0].set_aspect("equal")

    # -------------------------------------------------------------
    # Cols 1..4: Model Comparisons
    # -------------------------------------------------------------
    for col, k in enumerate(keys, start=1):
        model, is_umnn, test_ll, p_count = models_dict[k]

        # Row 1: Density Contours
        ax_top = axes[0, col]
        X, Y, prob = evaluate_density_grid(model, grid_res=90, range_lim=3.8, is_umnn=is_umnn)
        cf = ax_top.contourf(X, Y, prob, levels=35, cmap="magma")
        fig.colorbar(cf, ax=ax_top, fraction=0.046, pad=0.04)
        ax_top.set_title(f"{k}\nTest LL: {test_ll:+.3f} nats | {p_count:,} params", fontsize=11, fontweight="bold")
        ax_top.set_xlim(-4, 4)
        ax_top.set_ylim(-4, 4)
        ax_top.set_aspect("equal")

        # Row 2: Generated Samples
        ax_bot = axes[1, col]
        samples = generate_samples(model, n_samples=1000, is_umnn=is_umnn)
        c = colors[col - 1]
        ax_bot.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.5, color=c)
        ax_bot.set_title(f"{k}\nGenerated Samples (N=1,000)", fontsize=10)
        ax_bot.set_xlim(-4, 4)
        ax_bot.set_ylim(-4, 4)
        ax_bot.set_aspect("equal")

    plt.suptitle(
        f"High-Capacity Continuous Density Estimation Benchmark: {dataset_name.upper()} (Parameter-Matched)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Successfully generated and saved: {save_path}")


def run_all_high_param_benchmarks(output_dir: str = "results/kan_benchmarks"):
    os.makedirs(output_dir, exist_ok=True)
    datasets = ["moons", "8gaussians", "pinwheel"]

    for dname in datasets:
        print(f"\n=======================================================")
        print(f"  Training High-Parameter Models on: {dname.upper()}")
        print(f"=======================================================")

        tr, va, te, test_tensor, full_data = get_2d_dataloaders(
            dataset_name=dname,
            n_train=4000,
            n_val=1000,
            n_test=1000,
            batch_size=256,
            seed=42,
        )

        models_eval = {}

        # 1. SM-KAN Compact (356 params)
        print("--> 1/4 Training SM-KAN (Compact, 1-step, 356 params)...")
        m_compact = SMKANDensityEstimator(num_features=2, num_steps=1, hidden_dim=6, grid_size=8)
        m_compact = train_model(m_compact, tr, va, epochs=15, lr=8e-3)
        with torch.no_grad():
            ll_compact = m_compact.log_prob(test_tensor).mean().item()
        models_eval["SM-KAN (Compact)"] = (m_compact, False, ll_compact, count_parameters(m_compact))

        # 2. SM-KAN High-Capacity (4-step, 24,488 params - matched to RealNVP)
        print("--> 2/4 Training SM-KAN (High-Capacity, 4-step, 24,488 params)...")
        m_high = SMKANDensityEstimator(num_features=2, num_steps=4, hidden_dim=64, grid_size=16)
        m_high = train_model(m_high, tr, va, epochs=15, lr=3e-3)
        with torch.no_grad():
            ll_high = m_high.log_prob(test_tensor).mean().item()
        models_eval["SM-KAN (High-Cap)"] = (m_high, False, ll_high, count_parameters(m_high))

        # 3. RealNVP Baseline (26,508 params)
        print("--> 3/4 Training RealNVP Baseline (26,508 params)...")
        m_nvp = RealNVPFlow(num_features=2, num_layers=6, hidden_dim=64)
        m_nvp = train_model(m_nvp, tr, va, epochs=15, lr=3e-3)
        with torch.no_grad():
            ll_nvp = m_nvp.log_prob(test_tensor).mean().item()
        models_eval["RealNVP"] = (m_nvp, False, ll_nvp, count_parameters(m_nvp))

        # 4. UMNN-MAF Baseline (5,473 params)
        print("--> 4/4 Training UMNN-MAF Baseline (5,473 params)...")
        m_umnn = UMNNMAFFlow(nb_flow=1, nb_in=2, hidden_derivative=[40, 40], hidden_embedding=[40, 40], embedding_s=16, nb_steps=20)
        m_umnn = train_model(m_umnn, tr, va, epochs=10, lr=3e-3, is_umnn=True)
        with torch.no_grad():
            ll_umnn, _ = m_umnn.compute_ll(test_tensor)
            ll_umnn = ll_umnn.mean().item()
        models_eval["UMNN-MAF"] = (m_umnn, True, ll_umnn, count_parameters(m_umnn))

        # Save figure
        fig_path = os.path.join(output_dir, f"density_high_param_{dname}.png")
        create_high_param_comparison_figure(
            dataset_name=dname,
            target_data=full_data[:2000],
            models_dict=models_eval,
            save_path=fig_path,
        )


if __name__ == "__main__":
    run_all_high_param_benchmarks()
