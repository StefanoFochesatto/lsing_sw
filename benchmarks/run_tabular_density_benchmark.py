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
from src.datasets.uci_loader import get_tabular_dataloaders


PUBLISHED_SOTA = {
    "POWER": {
        "RealNVP": 0.17,
        "MAF": 0.24,
        "Glow": 0.42,
        "FFJORD": 0.46,
        "UMNN-MAF": 0.63,
        "RQ-NSF (C)": 0.64,
        "RQ-NSF (AR)": 0.66,
        "TraDE": 0.73,
    },
    "GAS": {
        "RealNVP": 8.33,
        "MAF": 10.08,
        "Glow": 12.24,
        "FFJORD": 8.59,
        "UMNN-MAF": 10.89,
        "RQ-NSF (C)": 12.80,
        "RQ-NSF (AR)": 13.09,
        "TraDE": 13.27,
    },
    "HEPMASS": {
        "RealNVP": -18.71,
        "MAF": -17.70,
        "Glow": -16.99,
        "FFJORD": -14.92,
        "UMNN-MAF": -13.99,
        "RQ-NSF (C)": -14.27,
        "RQ-NSF (AR)": -14.01,
        "TraDE": -12.01,
    },
}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_tabular_model(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    device: str,
) -> Tuple[nn.Module, float]:
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)

    best_val_loss = float("inf")
    best_state = None

    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        for (batch,) in train_loader:
            batch = batch.to(device)
            loss = -model.log_prob(batch).mean()

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
                lp = model.log_prob(batch)
                val_loss += (-lp.mean().item())
                val_batches += 1
        val_loss /= max(val_batches, 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return model, train_time


def evaluate_tabular_model(
    model: nn.Module,
    test_tensor: torch.Tensor,
    device: str,
    n_sample_eval: int = 1000,
) -> Tuple[float, float, float]:
    model.eval()
    test_tensor = test_tensor.to(device)

    # Evaluate test log-likelihood in chunks
    chunk_size = 1000
    lps = []
    with torch.no_grad():
        for i in range(0, test_tensor.shape[0], chunk_size):
            chunk = test_tensor[i : i + chunk_size]
            lps.append(model.log_prob(chunk))
        full_lp = torch.cat(lps, dim=0)
        test_ll = full_lp.mean().item()

    # Benchmark generative sampling speed
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model.sample(n_sample_eval, device=device)
    sample_time = time.perf_counter() - t0
    sample_rate = float(n_sample_eval) / max(sample_time, 1e-6)

    return float(test_ll), float(sample_time), float(sample_rate)


def print_comparison_table(dataset_name: str, results: Dict):
    dkey = dataset_name.upper()
    print(f"\n=========================================================================================")
    print(f"  BENCHMARK SUMMARY: {dkey} TABULAR DENSITY ESTIMATION")
    print(f"=========================================================================================")
    print(f"{'Model':<20} | {'Test LL (nats)':<15} | {'Params':<10} | {'Train Time (s)':<15} | {'Samples/sec':<12}")
    print("-" * 85)

    # Evaluated models
    if "sm_kan" in results:
        r = results["sm_kan"]
        print(f"{'SM-KAN (Ours)':<20} | {r['test_ll']:<15.3f} | {r['params']:<10} | {r['train_time_sec']:<15.2f} | {r['sample_rate_hz']:<12.1f}")
    if "real_nvp" in results:
        r = results["real_nvp"]
        print(f"{'RealNVP':<20} | {r['test_ll']:<15.3f} | {r['params']:<10} | {r['train_time_sec']:<15.2f} | {r['sample_rate_hz']:<12.1f}")

    # Published SOTA references
    if dkey in PUBLISHED_SOTA:
        print("-" * 85)
        print("  [Published SOTA Reference Numbers (Papamakarios et al., Durkan et al.)]:")
        for mname, sota_ll in PUBLISHED_SOTA[dkey].items():
            print(f"  * {mname:<18} : {sota_ll:+.2f} nats")
    print("=" * 85)


def run_tabular_benchmark(
    datasets: List[str],
    data_dir: str = "data",
    epochs: int = 15,
    batch_size: int = 512,
    lr: float = 8e-3,
    subsample: int = 10000,
    hidden_dim: int = 4,
    grid_size: int = 8,
    device: str = "cpu",
    output_dir: str = "results/kan_benchmarks",
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    for dname in datasets:
        print(f"\n=======================================================")
        print(f"  Running Tabular Density Benchmark on: {dname.upper()}")
        print(f"=======================================================")

        train_loader, val_loader, test_loader, test_tensor, num_features, source_type = get_tabular_dataloaders(
            dataset_name=dname,
            data_dir=data_dir,
            batch_size=batch_size,
            subsample_train=subsample,
            seed=42,
        )
        print(f"Loaded {dname.upper()} (Source: {source_type}): {num_features} dimensions, {len(train_loader.dataset)} training samples.")

        dataset_results = {
            "num_features": num_features,
            "source_type": source_type,
            "n_train": len(train_loader.dataset),
            "n_test": len(test_tensor),
        }

        # 1. SM-KAN
        print("\n--> Training SM-KAN (Ours)...")
        smkan = SMKANDensityEstimator(
            num_features=num_features,
            hidden_dim=hidden_dim,
            grid_size=grid_size,
            conditioning_scale=True,
            dev=device,
        )
        smkan, train_time_kan = train_tabular_model(smkan, train_loader, val_loader, epochs=epochs, lr=lr, device=device)
        test_ll_kan, sample_time_kan, sample_rate_kan = evaluate_tabular_model(smkan, test_tensor, device=device)

        dataset_results["sm_kan"] = {
            "test_ll": test_ll_kan,
            "train_time_sec": train_time_kan,
            "sample_time_1000": sample_time_kan,
            "sample_rate_hz": sample_rate_kan,
            "params": count_parameters(smkan),
        }
        print(f"    SM-KAN Test LL: {test_ll_kan:.3f} nats | Train Time: {train_time_kan:.2f}s | Sampling Speed: {sample_rate_kan:.1f} samp/s")

        # 2. RealNVP
        print("\n--> Training RealNVP Baseline...")
        realnvp = RealNVPFlow(num_features=num_features, num_layers=6, hidden_dim=64)
        realnvp, train_time_nvp = train_tabular_model(realnvp, train_loader, val_loader, epochs=epochs, lr=lr * 0.5, device=device)
        test_ll_nvp, sample_time_nvp, sample_rate_nvp = evaluate_tabular_model(realnvp, test_tensor, device=device)

        dataset_results["real_nvp"] = {
            "test_ll": test_ll_nvp,
            "train_time_sec": train_time_nvp,
            "sample_time_1000": sample_time_nvp,
            "sample_rate_hz": sample_rate_nvp,
            "params": count_parameters(realnvp),
        }
        print(f"    RealNVP Test LL: {test_ll_nvp:.3f} nats | Train Time: {train_time_nvp:.2f}s | Sampling Speed: {sample_rate_nvp:.1f} samp/s")

        all_results[dname] = dataset_results
        print_comparison_table(dname, dataset_results)

    json_path = os.path.join(output_dir, "tabular_density_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll tabular benchmark results saved to: {json_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run Tabular Density Estimation Benchmark (POWER, GAS)")
    parser.add_argument("--datasets", nargs="+", default=["power", "gas"], help="Tabular datasets to benchmark")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory where data is stored")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=8e-3, help="Learning rate")
    parser.add_argument("--subsample", type=int, default=10000, help="Max training samples for fast benchmark")
    parser.add_argument("--hidden_dim", type=int, default=4, help="Hidden dimension Q for KAN")
    parser.add_argument("--grid_size", type=int, default=8, help="B-spline grid intervals")
    parser.add_argument("--fast", action="store_true", help="Fast run with fewer epochs and samples")
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch device")
    args = parser.parse_args()

    epochs = 5 if args.fast else args.epochs
    subsample = 3000 if args.fast else args.subsample

    run_tabular_benchmark(
        datasets=args.datasets,
        data_dir=args.data_dir,
        epochs=epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        subsample=subsample,
        hidden_dim=args.hidden_dim,
        grid_size=args.grid_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
