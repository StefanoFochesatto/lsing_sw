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
from tqdm import tqdm

from models.sm_kan_density import SMKANDensityEstimator
from models.real_nvp import RealNVPFlow
from models.UMNN.UMNNMAFFlow import UMNNMAFFlow
from src.datasets.synthetic_2d import get_2d_dataloaders


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(
    model_name: str,
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    device: str,
) -> Tuple[nn.Module, float]:
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)

    best_val_loss = float("inf")
    best_state = None

    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        for (batch,) in train_loader:
            batch = batch.to(device)
            if model_name == "sm_kan":
                loss = -model.log_prob(batch).mean()
            elif model_name == "real_nvp":
                loss = -model.log_prob(batch).mean()
            elif model_name == "umnn":
                ll, _ = model.compute_ll(batch)
                loss = -ll.mean()
            else:
                raise ValueError(f"Unknown model_name: {model_name}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                if model_name in ["sm_kan", "real_nvp"]:
                    lp = model.log_prob(batch)
                    val_loss += (-lp.mean().item())
                elif model_name == "umnn":
                    ll, _ = model.compute_ll(batch)
                    val_loss += (-ll.mean().item())
                val_batches += 1
        val_loss /= max(val_batches, 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return model, train_time


def evaluate_model(
    model_name: str,
    model: nn.Module,
    test_tensor: torch.Tensor,
    device: str,
    n_sample_eval: int = 1000,
) -> Tuple[float, float, torch.Tensor]:
    model.eval()
    test_tensor = test_tensor.to(device)

    # Test log-likelihood
    with torch.no_grad():
        if model_name in ["sm_kan", "real_nvp"]:
            test_ll = model.log_prob(test_tensor).mean().item()
        elif model_name == "umnn":
            ll, _ = model.compute_ll(test_tensor)
            test_ll = ll.mean().item()

    # Sampling time
    t0 = time.perf_counter()
    with torch.no_grad():
        if model_name == "sm_kan":
            samples = model.sample(n_sample_eval, device=device, tol=1e-5)
        elif model_name == "real_nvp":
            samples = model.sample(n_sample_eval, device=device)
        elif model_name == "umnn":
            z = torch.randn(n_sample_eval, 2, device=device)
            # Suppress print statements inside UMNN invert
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                samples = model.invert(z, iter=15)
            finally:
                sys.stdout = old_stdout
    sample_time = time.perf_counter() - t0

    return test_ll, sample_time, samples.cpu()


def compute_density_grid(
    model_name: str,
    model: nn.Module,
    grid_res: int = 100,
    range_lim: float = 4.0,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = torch.linspace(-range_lim, range_lim, grid_res)
    ys = torch.linspace(-range_lim, range_lim, grid_res)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    pts = torch.stack([X.flatten(), Y.flatten()], dim=-1).to(device)

    model.eval()
    with torch.no_grad():
        # Evaluate in chunks to prevent memory spikes
        chunk_size = 2000
        lps = []
        for i in range(0, pts.shape[0], chunk_size):
            chunk = pts[i : i + chunk_size]
            if model_name in ["sm_kan", "real_nvp"]:
                lp = model.log_prob(chunk)
            elif model_name == "umnn":
                lp, _ = model.compute_ll(chunk)
            lps.append(lp.cpu())
        lp_full = torch.cat(lps, dim=0)
        prob = torch.exp(lp_full).reshape(grid_res, grid_res).numpy()

    return X.numpy(), Y.numpy(), prob


def plot_side_by_side(
    dataset_name: str,
    target_data: torch.Tensor,
    models_dict: Dict[str, nn.Module],
    samples_dict: Dict[str, torch.Tensor],
    metrics_dict: Dict[str, Dict],
    save_path: str,
):
    model_keys = ["sm_kan", "real_nvp", "umnn"]
    model_names = ["SM-KAN (Ours)", "RealNVP", "UMNN-MAF"]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    # Row 1: Density Contours
    # Col 0: Ground Truth Target Data
    axes[0, 0].scatter(target_data[:, 0], target_data[:, 1], s=2, alpha=0.3, color="#1f77b4")
    axes[0, 0].set_title(f"Target Distribution: {dataset_name.upper()}", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlim(-4, 4)
    axes[0, 0].set_ylim(-4, 4)
    axes[0, 0].set_aspect("equal")

    for col, (k, title) in enumerate(zip(model_keys, model_names), start=1):
        ax = axes[0, col]
        X, Y, prob = compute_density_grid(k, models_dict[k], grid_res=80, range_lim=4.0)
        cf = ax.contourf(X, Y, prob, levels=30, cmap="viridis")
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        test_ll = metrics_dict[k]["test_ll"]
        ax.set_title(f"{title} Density\nTest LL: {test_ll:.2f} nats", fontsize=11, fontweight="bold")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect("equal")

    # Row 2: Generated Samples
    axes[1, 0].scatter(target_data[:1000, 0], target_data[:1000, 1], s=4, alpha=0.5, color="#1f77b4")
    axes[1, 0].set_title("Ground Truth Samples", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlim(-4, 4)
    axes[1, 0].set_ylim(-4, 4)
    axes[1, 0].set_aspect("equal")

    colors = ["#2ca02c", "#ff7f0e", "#9467bd"]
    for col, (k, title, c) in enumerate(zip(model_keys, model_names, colors), start=1):
        ax = axes[1, col]
        samp = samples_dict[k].numpy()
        ax.scatter(samp[:, 0], samp[:, 1], s=4, alpha=0.5, color=c)
        s_time = metrics_dict[k]["sample_time_1000"]
        p_count = metrics_dict[k]["params"]
        ax.set_title(f"{title} Generated Samples\n1k samples: {s_time*1000:.1f}ms | {p_count} params", fontsize=10)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect("equal")

    plt.suptitle(
        f"2D Continuous Density Estimation Benchmark: {dataset_name.upper()} (SM-KAN vs RealNVP vs UMNN)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved benchmark figure to: {save_path}")


def run_benchmark(
    datasets: List[str],
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 5e-3,
    device: str = "cpu",
    output_dir: str = "results/kan_benchmarks",
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    for dname in datasets:
        print(f"\n=======================================================")
        print(f"  Running 2D Density Benchmark on: {dname.upper()}")
        print(f"=======================================================")

        train_loader, val_loader, test_loader, test_tensor, full_data = get_2d_dataloaders(
            dataset_name=dname,
            n_train=6000,
            n_val=1500,
            n_test=1500,
            batch_size=batch_size,
            seed=42,
        )

        dataset_results = {}
        trained_models = {}
        generated_samples = {}

        # 1. SM-KAN
        print("--> Training SM-KAN (Ours)...")
        smkan = SMKANDensityEstimator(num_features=2, hidden_dim=6, grid_size=8, conditioning_scale=True, dev=device)
        smkan, train_time_kan = train_model("sm_kan", smkan, train_loader, val_loader, epochs=epochs, lr=lr, device=device)
        test_ll_kan, sample_time_kan, samp_kan = evaluate_model("sm_kan", smkan, test_tensor, device=device)

        dataset_results["sm_kan"] = {
            "test_ll": float(test_ll_kan),
            "train_time_sec": float(train_time_kan),
            "sample_time_1000": float(sample_time_kan),
            "params": int(count_parameters(smkan)),
        }
        trained_models["sm_kan"] = smkan
        generated_samples["sm_kan"] = samp_kan
        print(f"    SM-KAN Test LL: {test_ll_kan:.3f} nats | Train Time: {train_time_kan:.2f}s | 1k Sample Time: {sample_time_kan*1000:.1f}ms")

        # 2. RealNVP
        print("--> Training RealNVP...")
        realnvp = RealNVPFlow(num_features=2, num_layers=6, hidden_dim=64)
        realnvp, train_time_nvp = train_model("real_nvp", realnvp, train_loader, val_loader, epochs=epochs, lr=lr * 0.5, device=device)
        test_ll_nvp, sample_time_nvp, samp_nvp = evaluate_model("real_nvp", realnvp, test_tensor, device=device)

        dataset_results["real_nvp"] = {
            "test_ll": float(test_ll_nvp),
            "train_time_sec": float(train_time_nvp),
            "sample_time_1000": float(sample_time_nvp),
            "params": int(count_parameters(realnvp)),
        }
        trained_models["real_nvp"] = realnvp
        generated_samples["real_nvp"] = samp_nvp
        print(f"    RealNVP Test LL: {test_ll_nvp:.3f} nats | Train Time: {train_time_nvp:.2f}s | 1k Sample Time: {sample_time_nvp*1000:.1f}ms")

        # 3. UMNN
        print("--> Training UMNN-MAF...")
        umnn = UMNNMAFFlow(nb_flow=1, nb_in=2, hidden_derivative=[40, 40], hidden_embedding=[40, 40], embedding_s=16, nb_steps=20)
        umnn, train_time_umnn = train_model("umnn", umnn, train_loader, val_loader, epochs=epochs, lr=lr * 0.5, device=device)
        test_ll_umnn, sample_time_umnn, samp_umnn = evaluate_model("umnn", umnn, test_tensor, device=device)

        dataset_results["umnn"] = {
            "test_ll": float(test_ll_umnn),
            "train_time_sec": float(train_time_umnn),
            "sample_time_1000": float(sample_time_umnn),
            "params": int(count_parameters(umnn)),
        }
        trained_models["umnn"] = umnn
        generated_samples["umnn"] = samp_umnn
        print(f"    UMNN Test LL: {test_ll_umnn:.3f} nats | Train Time: {train_time_umnn:.2f}s | 1k Sample Time: {sample_time_umnn*1000:.1f}ms")

        all_results[dname] = dataset_results

        # Generate side-by-side plots
        fig_path = os.path.join(output_dir, f"density_2d_{dname}.png")
        plot_side_by_side(
            dataset_name=dname,
            target_data=full_data[:2000],
            models_dict=trained_models,
            samples_dict=generated_samples,
            metrics_dict=dataset_results,
            save_path=fig_path,
        )

    json_path = os.path.join(output_dir, "density_2d_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll 2D benchmark results saved to: {json_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run 2D Density Estimation Benchmark")
    parser.add_argument("--datasets", nargs="+", default=["moons", "8gaussians", "pinwheel"], help="Datasets to evaluate")
    parser.add_argument("--epochs", type=int, default=35, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    parser.add_argument("--fast", action="store_true", help="Fast run with fewer epochs")
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch device")
    args = parser.parse_args()

    epochs = 10 if args.fast else args.epochs
    run_benchmark(
        datasets=args.datasets,
        epochs=epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()
