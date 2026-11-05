import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union


def invert_coordinate_map(
    map_k: nn.Module,
    z_k: torch.Tensor,
    x_cond: Optional[torch.Tensor] = None,
    max_iter: int = 30,
    tol: float = 1e-6,
    initial_guess: Optional[torch.Tensor] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    Vectorized root-finding for S^k(x_{<k}, x_k) = z_k using batched Newton-Raphson
    with bisection safeguarding.

    Given fixed conditioning values x_{<k} and target latent values z_k, solves:
        f(x_k) = S^k(x_{<k}, x_k) - z_k = 0

    Because partial_k S^k(x_{<k}, x_k) >= e^{-s_max} * eps > 0 strictly everywhere on R,
    f is strictly monotonic with a unique root for every sample in the batch.

    Args:
        map_k: MonotonicKANMap or ScaleModulatedKANMap instance for component k.
        z_k: Latent targets of shape (B, 1) or (B,).
        x_cond: Conditioning features x_{<k} of shape (B, k) or None if k=0.
        max_iter: Maximum Newton-Raphson iterations (typically converges in 3-5).
        tol: Residual convergence tolerance ||S^k(x) - z_k||_inf < tol.
        initial_guess: Optional starting tensor of shape (B, 1) or (B,).
        device: Target PyTorch device.

    Returns:
        x_k: Inverted values of shape (B, 1).
        diagnostics: Dictionary containing convergence flag, iteration count,
                     final maximum residual, and residual history.
    """
    if z_k.dim() == 1:
        z_k = z_k.unsqueeze(1)

    B = z_k.shape[0]
    dev = z_k.device if device is None else torch.device(device)
    dtype = z_k.dtype
    z_k = z_k.to(device=dev)

    if x_cond is not None:
        x_cond = x_cond.to(device=dev, dtype=dtype)

    eps_float = 1e-5 if dtype == torch.float32 else 1e-11
    effective_tol = max(tol, eps_float)

    if initial_guess is not None:
        if initial_guess.dim() == 1:
            x = initial_guess.unsqueeze(1).clone().to(device=dev, dtype=dtype)
        else:
            x = initial_guess.clone().to(device=dev, dtype=dtype)
    else:
        # Default initialization: since maps are initialized near identity,
        # z_k is an excellent starting point.
        x = z_k.clone()

    # Brackets for bisection safeguarding: [a, b] such that f(a) <= 0 <= f(b)
    a = torch.full((B, 1), -float("inf"), device=dev, dtype=dtype)
    b = torch.full((B, 1), float("inf"), device=dev, dtype=dtype)

    residual_history: List[float] = []
    prev_err = torch.full((B, 1), float("inf"), device=dev, dtype=dtype)

    for it in range(max_iter):
        with torch.no_grad():
            derivs = map_k.compute_derivatives(x, x_cond)
            f_val = derivs["out"] - z_k
            f_prime = derivs["S_k"].clamp(min=1e-7)

            err = f_val.abs()
            max_err = err.max().item()
            residual_history.append(max_err)

            if max_err <= effective_tol:
                break

            # Update brackets
            mask_neg = f_val < 0
            mask_pos = f_val > 0
            a = torch.where(mask_neg, torch.maximum(a, x), a)
            b = torch.where(mask_pos, torch.minimum(b, x), b)

            # Proposed standard Newton step
            step = f_val / f_prime
            x_newton = x - step

            # Safeguarding check: Newton step must fall strictly inside known bracket
            has_brackets = torch.isfinite(a) & torch.isfinite(b)
            inside_bracket = (x_newton > a) & (x_newton < b)

            # Bisection fallback if brackets are finite and Newton overshoots
            x_bisect = 0.5 * (a + b)

            # Clamped step when bracket is not yet finite on both sides
            step_clamped = torch.clamp(step, -5.0, 5.0)
            x_damped = x - step_clamped

            # Accept Newton if inside bracket and residual isn't blowing up
            use_newton = inside_bracket & (~has_brackets | (err <= prev_err * 1.5))
            use_bisect = has_brackets & (~use_newton)

            x_next = torch.where(
                use_newton,
                x_newton,
                torch.where(use_bisect, x_bisect, x_damped),
            )

            # Only update samples that have not yet reached tolerance
            not_done = err > effective_tol
            x = torch.where(not_done, x_next, x)
            prev_err = err

    with torch.no_grad():
        final_derivs = map_k.compute_derivatives(x, x_cond)
        final_res = (final_derivs["out"] - z_k).abs()
        final_max_err = final_res.max().item()
        converged = final_res.squeeze(-1) <= (effective_tol * 2.0)

    return x, {
        "converged": converged,
        "num_iters": it + 1,
        "final_max_residual": final_max_err,
        "residual_history": residual_history,
    }


def invert_triangular_map(
    maps: Union[nn.ModuleList, List[nn.Module]],
    z: torch.Tensor,
    max_iter: int = 30,
    tol: float = 1e-6,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[torch.Tensor, List[Dict]]:
    """
    Invert a full lower-triangular Knothe-Rosenblatt map S: R^d -> R^d coordinate-by-coordinate.

    Args:
        maps: List or ModuleList of d component maps S^k, where map k takes inputs of dimension k+1.
        z: Latent batch tensor of shape (B, d).
        max_iter: Maximum iterations per coordinate inversion.
        tol: Residual tolerance.
        device: Device to perform inversion on.

    Returns:
        x: Inverted samples in data space of shape (B, d).
        diagnostics: List of diagnostic dictionaries for each coordinate map.
    """
    B, d = z.shape
    dev = z.device if device is None else torch.device(device)
    dtype = z.dtype

    x = torch.empty((B, d), device=dev, dtype=dtype)
    diagnostics = []

    for k in range(d):
        map_k = maps[k].to(dev)
        z_k = z[:, [k]]
        x_cond = x[:, :k] if k > 0 else None

        x_k, diag_k = invert_coordinate_map(
            map_k=map_k,
            z_k=z_k,
            x_cond=x_cond,
            max_iter=max_iter,
            tol=tol,
            device=dev,
        )
        x[:, [k]] = x_k
        diagnostics.append(diag_k)

    return x, diagnostics
