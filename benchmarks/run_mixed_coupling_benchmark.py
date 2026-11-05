#!/usr/bin/env python3
"""
Benchmark: Scale-Modulated Monotonic KAN (SM-KAN) vs. Additive Monotonic KAN
on the 6-Dimensional Mixed Additive & Multiplicative Coupling Graph.

Verifies three simultaneous regimes in one graphical model:
    1. Purely Additive Pair (0, 1):        X_1 = 1.2 sin(X_0) + Z_1
    2. Purely Multiplicative Pair (2, 3):  X_3 = X_2 * Z_3
    3. Mixed Add + Mult Node 5 (4, 5; 2, 5): X_5 = 1.2 X_4 + X_2 * Z_5
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

from models.monotonic_kan import BSplineBasis, MonotonicKANMap, ScaleModulatedKANMap
from trainers.kan_trainer import KANTrainer, KANPrecisionComputer
from datasets.mixed_coupling_generator import (
    generate_mixed_coupling_dataset,
    get_mixed_coupling_ground_truth,
)


def inspect_pathway_activations(sm_model: ScaleModulatedKANMap, data: torch.Tensor) -> dict:
    """
    Computes empirical standard deviations of the learned scale splines psi_j(X_j)
    and offset splines rho_j(X_j) across evaluation samples.
    """
    others = sm_model.other_indices
    with torch.no_grad():
        b_val, _, _, _ = BSplineBasis.eval_basis(data, sm_model.grid, sm_model.spline_order)
        b_cond = b_val[:, others, :]
        psi_vals = torch.einsum("bdm,dm->bd", b_cond, sm_model.scale_weights)
        rho_vals = torch.einsum("bdm,dm->bd", b_cond, sm_model.offset_weights)

        stats = {}
        for idx, j in enumerate(others):
            stats[f"X_{j}"] = {
                "std_psi_scale": float(psi_vals[:, idx].std().item()),
                "std_rho_offset": float(rho_vals[:, idx].std().item()),
            }
        return stats


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    output_dir = REPO_ROOT / "results/kan_benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MIXED ADDITIVE & MULTIPLICATIVE COUPLING BENCHMARK (d=6)")
    print("=" * 80)

    n_trn, n_val, n_tst = 2500, 1000, 1500
    full_data = generate_mixed_coupling_dataset(n_trn + n_val + n_tst, standardize=True, seed=42)
    trn_data = full_data[:n_trn]
    val_data = full_data[n_trn : n_trn + n_val]
    tst_data = full_data[n_trn + n_val :]

    d = 6
    true_adj = get_mixed_coupling_ground_truth()
    regs = [0.005]
    max_epochs = 35

    # 1. Additive Monotonic KAN
    print("\n[1/2] Training Additive MonotonicKANMap...")
    t0 = time.perf_counter()
    base_add = [
        MonotonicKANMap(d, target_index=k, hidden_dim=4, grid_size=8, conditioning_scale=False)
        for k in range(d)
    ]
    trainer_add = KANTrainer(base_add, d, hidden_dim=4, grid_size=8, max_epochs=max_epochs)
    add_models = []
    add_nlls = {}
    for k in range(d):
        others = [j for j in range(d) if j != k]
        best_m, _, _, _ = trainer_add.train_model(trn_data, k, others, regs, val_data)
        add_models.append(best_m)
        _, nll_k, _ = best_m.compute_loss(tst_data)
        add_nlls[f"node_{k}"] = float(nll_k.item())
    add_train_time = time.perf_counter() - t0

    comp_add = KANPrecisionComputer(add_models, tst_data, d)
    raw_add_asym = comp_add.compute()
    raw_add = 0.5 * (raw_add_asym + raw_add_asym.T)
    np.fill_diagonal(raw_add, 1.0)
    norm_add = comp_add.make_symmetric(raw_add_asym)

    # 2. Scale-Modulated Monotonic KAN
    print("[2/2] Training ScaleModulatedKANMap...")
    t0 = time.perf_counter()
    base_sm = [ScaleModulatedKANMap(d, target_index=k, hidden_dim=4, grid_size=8) for k in range(d)]
    trainer_sm = KANTrainer(base_sm, d, hidden_dim=4, grid_size=8, max_epochs=max_epochs)
    sm_models = []
    sm_nlls = {}
    for k in range(d):
        others = [j for j in range(d) if j != k]
        best_m, _, _, _ = trainer_sm.train_model(trn_data, k, others, regs, val_data)
        sm_models.append(best_m)
        _, nll_k, _ = best_m.compute_loss(tst_data)
        sm_nlls[f"node_{k}"] = float(nll_k.item())
    sm_train_time = time.perf_counter() - t0

    comp_sm = KANPrecisionComputer(sm_models, tst_data, d)
    raw_sm_asym = comp_sm.compute()
    raw_sm = 0.5 * (raw_sm_asym + raw_sm_asym.T)
    np.fill_diagonal(raw_sm, 1.0)
    norm_sm = comp_sm.make_symmetric(raw_sm_asym)

    pathway_stats = {
        "node_1_pure_additive": inspect_pathway_activations(sm_models[1], tst_data),
        "node_3_pure_multiplicative": inspect_pathway_activations(sm_models[3], tst_data),
        "node_5_mixed_add_mult": inspect_pathway_activations(sm_models[5], tst_data),
    }

    results = {
        "ground_truth_adj": true_adj.tolist(),
        "AdditiveMonotonicKAN": {
            "train_time_sec": add_train_time,
            "test_nll_per_node": add_nlls,
            "raw_precision_matrix": raw_add.tolist(),
            "normalized_precision_matrix": norm_add.tolist(),
        },
        "ScaleModulatedKAN": {
            "train_time_sec": sm_train_time,
            "test_nll_per_node": sm_nlls,
            "raw_precision_matrix": raw_sm.tolist(),
            "normalized_precision_matrix": norm_sm.tolist(),
            "pathway_spline_std": pathway_stats,
        },
    }

    out_json = output_dir / "mixed_coupling_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print("\n--- Summary of Raw Symmetrized Precision Entries ---")
    print(f"  Purely Additive Edge (0, 1):       Additive KAN = {raw_add[0, 1]:.3f} | SM-KAN = {raw_sm[0, 1]:.3f}")
    print(f"  Purely Multiplicative Edge (2, 3): Additive KAN = {raw_add[2, 3]:.3f} | SM-KAN = {raw_sm[2, 3]:.3f}")
    print(f"  Mixed Additive Edge (4, 5):        Additive KAN = {raw_add[4, 5]:.3f} | SM-KAN = {raw_sm[4, 5]:.3f}")
    print(f"  Mixed Multiplicative Edge (2, 5):  Additive KAN = {raw_add[2, 5]:.3f} | SM-KAN = {raw_sm[2, 5]:.3f}")
    print(f"\nResults saved to {out_json}")


if __name__ == "__main__":
    main()
