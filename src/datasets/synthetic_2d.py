from typing import Tuple
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def make_2d_dataset(
    dataset_name: str,
    n_samples: int = 10000,
    seed: int = 42,
    noise: float = 0.05,
) -> torch.Tensor:
    """
    Generate canonical 2D synthetic datasets for continuous density estimation.

    Supported distributions:
        - 'moons': Two interlocking non-convex crescent densities.
        - '8gaussians': Circular mixture of 8 isolated Gaussian modes with low-density basins.
        - 'pinwheel': Multi-spiral distribution with sharp non-linear curvature.
        - 'swissroll': 2D winding non-linear manifold.
        - 'checkerboard': Piecewise discontinuous 2x2 or 4x4 density grid.
    """
    rng = np.random.RandomState(seed)

    if dataset_name.lower() == "moons":
        n_half = n_samples // 2
        theta1 = rng.uniform(0, np.pi, n_half)
        u1 = np.stack([np.cos(theta1), np.sin(theta1)], axis=1) + rng.randn(n_half, 2) * noise
        u1[:, 0] -= 0.5
        u1[:, 1] += 0.25

        theta2 = rng.uniform(0, np.pi, n_samples - n_half)
        u2 = np.stack([1.0 - np.cos(theta2), 1.0 - np.sin(theta2) - 0.5], axis=1) + rng.randn(n_samples - n_half, 2) * noise
        u2[:, 0] -= 0.5
        u2[:, 1] += 0.25

        data = np.concatenate([u1, u2], axis=0)
        rng.shuffle(data)
        data = data * 2.0  # scale nicely to ~[-3, 3]

    elif dataset_name.lower() in ["8gaussians", "eight_gaussians"]:
        scale = 3.0
        centers = [
            (scale * np.cos(2.0 * np.pi * i / 8.0), scale * np.sin(2.0 * np.pi * i / 8.0))
            for i in range(8)
        ]
        indices = rng.choice(8, size=n_samples)
        centers_arr = np.array(centers)[indices]
        noise_arr = rng.randn(n_samples, 2) * 0.4
        data = centers_arr + noise_arr

    elif dataset_name.lower() == "pinwheel":
        num_classes = 5
        rate = 0.25
        rad = rng.rand(n_samples) * 0.65 + 0.25
        theta = 2.0 * np.pi * rng.randint(0, num_classes, n_samples) / float(num_classes) + rate * rad * 2.0 * np.pi
        x = rad * np.cos(theta) * 3.5 + rng.randn(n_samples) * 0.08
        y = rad * np.sin(theta) * 3.5 + rng.randn(n_samples) * 0.08
        data = np.stack([x, y], axis=1)

    elif dataset_name.lower() == "swissroll":
        theta = 1.5 * np.pi * (1.0 + 2.0 * rng.rand(n_samples))
        x = theta * np.cos(theta)
        y = theta * np.sin(theta)
        data = np.stack([x, y], axis=1) / 3.0 + rng.randn(n_samples, 2) * 0.1

    elif dataset_name.lower() == "checkerboard":
        x1 = rng.rand(n_samples * 2) * 4.0 - 2.0
        x2_rand = rng.rand(n_samples * 2) - rng.randint(0, 2, n_samples * 2) * 2.0
        x2 = x2_rand + (np.floor(x1) % 2)
        valid = (x1 >= -2.0) & (x1 <= 2.0) & (x2 >= -2.0) & (x2 <= 2.0)
        data = np.stack([x1[valid], x2[valid]], axis=1)[:n_samples] * 1.5

    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    return torch.tensor(data, dtype=torch.float32)


def get_2d_dataloaders(
    dataset_name: str,
    n_train: int = 10000,
    n_val: int = 2000,
    n_test: int = 2000,
    batch_size: int = 256,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    """
    Returns train, validation, and test DataLoaders along with test tensor and full dataset tensor.
    """
    total = n_train + n_val + n_test
    data = make_2d_dataset(dataset_name, n_samples=total, seed=seed)

    train_data = data[:n_train]
    val_data = data[n_train : n_train + n_val]
    test_data = data[n_train + n_val :]

    train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_data), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, test_data, data
