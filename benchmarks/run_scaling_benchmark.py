import argparse
import json
import os
import time
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim

from models.sm_kan_density import SMKANDensityEstimator
from models.real_nvp import RealNVPFlow
from models.UMNN.UMNNMAFFlow import UMNNMAFFlow
from src.datasets.synthetic_2d import get_2d_dataloaders
from src.datasets.uci_loader import get_tabular_dataloaders


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_eval_model(
    model: nn.Module,
    train_loader,
    val_loader,
    test_tensor: torch.Tensor,
    epochs: int = 20,
    lr: float = 8e-3,
    device: str = "cpu",
    is_umnn: bool = False,
) -> Tuple[float, float, float]:
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
            ll, _ = model.compute_ll(test_tensor)
            test_ll = ll.mean().item()
        else:
            chunk_size = 1000
            lps = []
            for i in range(0, test_tensor.shape[0], chunk_size):
                chunk = test_tensor[i : i + chunk_size]
                lps.append(model.log_prob(chunk))
            test_ll = torch.cat(lps, dim=0).mean().item()

    # Sampling speed
    t0 = time.perf_counter()
    with torch.no_grad():
        if is_umnn:
            import io, sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                _ = model.invert(torch.randn(500, test_tensor.shape[1], device=device), iter=15)
            finally:
                sys.stdout = old_stdout
            sample_time = (time.perf_counter() - t0) * 2.0  # scale to 1000
        else:
            _ = model.sample(1000, device=device)
            sample_time = time.perf_counter() - t0

    return float(test_ll), float(train_time), float(sample_time)


def run_scaling_benchmark(output_dir: str = "results/kan_benchmarks") -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    device = "cpu"
    results = {}

    print("=========================================================================================")
    print("  PARAMETER-SCALED BENCHMARK: MATCHING KAN SIZE/DEPTH TO BASELINES")
    print("=========================================================================================")

    # -------------------------------------------------------------
    # 1. Two Moons Benchmark (2D)
    # -------------------------------------------------------------
    print("\n--- [2D Two Moons] Scaling Width (Q), Resolution (G), and Depth (K) ---")
    tr, va, te, test_t, _ = get_2d_dataloaders("moons", n_train=4000, n_val=1000, n_test=1000, batch_size=256, seed=42)

    configs_moons = [
        ("SM-KAN (Compact, 1-step)", SMKANDensityEstimator(num_features=2, num_steps=1, hidden_dim=6, grid_size=8)),
        ("SM-KAN (Matched to UMNN, 2-step)", SMKANDensityEstimator(num_features=2, num_steps=2, hidden_dim=32, grid_size=16)),
        ("SM-KAN (Matched to RealNVP, 4-step)", SMKANDensityEstimator(num_features=2, num_steps=4, hidden_dim=64, grid_size=16)),
        ("UMNN-MAF Baseline", UMNNMAFFlow(nb_flow=1, nb_in=2, hidden_derivative=[40, 40], hidden_embedding=[40, 40], embedding_s=16, nb_steps=20)),
        ("RealNVP Baseline (6-layer)", RealNVPFlow(num_features=2, num_layers=6, hidden_dim=64)),
    ]

    moons_res = {}
    for name, model in configs_moons:
        p_count = count_parameters(model)
        is_umnn = isinstance(model, UMNNMAFFlow)
        lr = 3e-3 if "RealNVP" in name or "4-step" in name else 8e-3
        epochs = 15 if not is_umnn else 10
        print(f"  Training {name} ({p_count:,} params)...")
        ll, t_train, t_samp = train_eval_model(model, tr, va, test_t, epochs=epochs, lr=lr, device=device, is_umnn=is_umnn)
        moons_res[name] = {
            "params": p_count,
            "test_ll": ll,
            "train_time_sec": t_train,
            "sample_time_1000": t_samp,
        }
        print(f"    --> Test LL: {ll:.3f} nats | Params: {p_count:,} | 1k Sample Time: {t_samp*1000:.1f}ms")
    results["moons"] = moons_res

    # -------------------------------------------------------------
    # 2. Tabular GAS Benchmark (8D)
    # -------------------------------------------------------------
    print("\n--- [8D Tabular GAS] Scaling Width (Q) and Resolution (G) ---")
    tr_g, va_g, te_g, test_tg, n_dim, _ = get_tabular_dataloaders("gas", batch_size=512, subsample_train=3000, seed=42)

    configs_gas = [
        ("SM-KAN (Compact, Q=4, G=8)", SMKANDensityEstimator(num_features=8, num_steps=1, hidden_dim=4, grid_size=8)),
        ("SM-KAN (Matched to RealNVP, Q=32, G=16)", SMKANDensityEstimator(num_features=8, num_steps=1, hidden_dim=32, grid_size=16)),
        ("RealNVP Baseline (6-layer)", RealNVPFlow(num_features=8, num_layers=6, hidden_dim=64)),
    ]

    gas_res = {}
    for name, model in configs_gas:
        p_count = count_parameters(model)
        lr = 4e-3 if "RealNVP" in name else 8e-3
        print(f"  Training {name} ({p_count:,} params)...")
        ll, t_train, t_samp = train_eval_model(model, tr_g, va_g, test_tg, epochs=8, lr=lr, device=device, is_umnn=False)
        gas_res[name] = {
            "params": p_count,
            "test_ll": ll,
            "train_time_sec": t_train,
            "sample_time_1000": t_samp,
        }
        print(f"    --> Test LL: {ll:.3f} nats | Params: {p_count:,} | 1k Sample Time: {t_samp*1000:.1f}ms")
    results["gas"] = gas_res

    # Save to JSON
    json_path = os.path.join(output_dir, "scaling_benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nScaling benchmark results saved to: {json_path}")

    return results


if __name__ == "__main__":
    run_scaling_benchmark()
