import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn


class RealNVPCouplingLayer(nn.Module):
    """
    Standard affine coupling layer for RealNVP (Dinh et al., 2017).
    Partitions input x into identity and transformed subsets according to a binary mask:
        x_id = x * mask,  x_tr = x * (1 - mask)
        s, t = net(x_id)
        y_id = x_id
        y_tr = x_tr * exp(s) + t
    """

    def __init__(self, num_features: int, mask: torch.Tensor, hidden_dim: int = 64):
        super().__init__()
        self.num_features = num_features
        self.register_buffer("mask", mask.float())
        in_dim = int(mask.sum().item())
        out_dim = num_features - in_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 2 * out_dim),
        )
        # Initialize output layer to zero for identity map start
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_bool = self.mask.bool()
        x_id = x[:, mask_bool]
        x_tr = x[:, ~mask_bool]

        st = self.net(x_id)
        s, t = st.chunk(2, dim=-1)
        s = 2.0 * torch.tanh(s / 2.0)  # Stabilize scale

        y_tr = x_tr * torch.exp(s) + t

        y = torch.empty_like(x)
        y[:, mask_bool] = x_id
        y[:, ~mask_bool] = y_tr

        ldj = s.sum(dim=-1)
        return y, ldj

    def invert(self, y: torch.Tensor) -> torch.Tensor:
        mask_bool = self.mask.bool()
        y_id = y[:, mask_bool]
        y_tr = y[:, ~mask_bool]

        st = self.net(y_id)
        s, t = st.chunk(2, dim=-1)
        s = 2.0 * torch.tanh(s / 2.0)

        x_tr = (y_tr - t) * torch.exp(-s)

        x = torch.empty_like(y)
        x[:, mask_bool] = y_id
        x[:, ~mask_bool] = x_tr
        return x


class RealNVPFlow(nn.Module):
    """
    Complete RealNVP Normalizing Flow architecture with alternating binary masks.
    """

    def __init__(self, num_features: int = 2, num_layers: int = 6, hidden_dim: int = 64):
        super().__init__()
        self.num_features = num_features
        layers = []
        for i in range(num_layers):
            mask = torch.zeros(num_features)
            # Alternating binary partition
            if num_features == 2:
                mask[i % 2] = 1.0
            else:
                half = num_features // 2
                if i % 2 == 0:
                    mask[:half] = 1.0
                else:
                    mask[half:] = 1.0
            layers.append(RealNVPCouplingLayer(num_features, mask, hidden_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ldj_tot = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        cur = x
        for layer in self.layers:
            cur, ldj = layer(cur)
            ldj_tot = ldj_tot + ldj
        return cur, ldj_tot

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z, ldj = self.forward(x)
        d = self.num_features
        log_pz = -0.5 * (d * math.log(2.0 * math.pi) + torch.sum(z**2, dim=-1))
        return log_pz + ldj

    def sample(self, n_samples: int, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
        dev = self.layers[0].mask.device if device is None else torch.device(device)
        z = torch.randn(n_samples, self.num_features, device=dev)
        cur = z
        for layer in reversed(self.layers):
            cur = layer.invert(cur)
        return cur
