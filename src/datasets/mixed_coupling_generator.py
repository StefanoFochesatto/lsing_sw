import os
import argparse
import numpy as np
import torch


def generate_mixed_coupling_dataset(num_samples: int, standardize: bool = True, seed: int | None = None) -> torch.Tensor:
    """
    Generates a 6-dimensional Mixed Additive & Multiplicative Coupling benchmark dataset.

    Structural equations (1-indexed coordinates X_1..X_6, 0-indexed 0..5):
        - Purely Additive Pair (0, 1):
            X_0 ~ N(0, 1),  X_1 = 1.2 * sin(X_0) + Z_1
          Ideal map: S^1(x) = x_1 - 1.2 * sin(x_0)  (target x_1 is unscaled: exp(psi_0(x_0)) = 1)
        - Purely Multiplicative Butterfly Pair (2, 3):
            X_2 ~ N(0, 1),  X_3 = X_2 * Z_3
          Ideal map: S^3(x) = x_3 / |x_2| = exp(-log|x_2|) * x_3
        - Mixed Additive + Multiplicative Node 5 with additive parent 4 and multiplicative parent 2:
            X_4 ~ N(0, 1),  X_5 = 1.2 * X_4 + X_2 * Z_5
          Ideal map: S^5(x) = (x_5 - 1.2 * x_4) / |x_2| = exp(-log|x_2|) * (x_5 - 1.2 * x_4)

    Args:
        num_samples (int): Number of i.i.d. samples to generate.
        standardize (bool): Whether to standardize each column to zero mean and unit variance.
        seed (int | None): Optional random seed for reproducibility.

    Returns:
        torch.Tensor: Tensor of shape (num_samples, 6) with dtype float32.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((num_samples, 6))
    x = np.zeros((num_samples, 6), dtype=np.float32)

    # 1. Purely additive pair (0, 1)
    x[:, 0] = z[:, 0]
    x[:, 1] = 1.2 * np.sin(x[:, 0]) + z[:, 1]

    # 2. Purely multiplicative pair (2, 3)
    x[:, 2] = z[:, 2]
    x[:, 3] = x[:, 2] * z[:, 3]

    # 3. Mixed additive + multiplicative node 5 (additive parent 4, multiplicative parent 2)
    x[:, 4] = z[:, 4]
    x[:, 5] = 1.2 * x[:, 4] + x[:, 2] * z[:, 5]

    if standardize:
        x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-8)

    return torch.tensor(x, dtype=torch.float32)


def get_mixed_coupling_ground_truth() -> np.ndarray:
    """
    Returns the 6x6 ground-truth conditional dependence adjacency matrix (Markov graph).
    True edges:
        (0, 1): purely additive coupling
        (2, 3): purely multiplicative coupling
        (4, 5): additive parent-child coupling on mixed node 5
        (2, 5): multiplicative parent-child coupling on mixed node 5
        (2, 4): moral co-parent coupling induced by common child node 5
    """
    adj = np.zeros((6, 6), dtype=int)
    edges = [(0, 1), (2, 3), (4, 5), (2, 5), (2, 4)]
    for u, v in edges:
        adj[u, v] = adj[v, u] = 1
    return adj


def main():
    parser = argparse.ArgumentParser(description="Generate 6D Mixed Additive & Multiplicative Coupling datasets.")
    parser.add_argument("--dataset_type", type=str, required=True, choices=["trn", "val", "tst"])
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="../../data/mixed_coupling")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dataset = generate_mixed_coupling_dataset(args.num_samples, standardize=True, seed=args.seed)
    out_path = os.path.join(args.output_dir, f"mixed_{args.dataset_type}_{args.num_samples}_6d.txt")
    np.savetxt(out_path, dataset.numpy(), fmt="%.8f", delimiter=",")
    print(f"Saved {args.dataset_type} dataset ({args.num_samples}x6) to {out_path}")


if __name__ == "__main__":
    main()
