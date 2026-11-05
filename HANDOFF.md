# Monotonic KAN Transport Map Project Handoff

## 1. Context & Repository Overview
We are working in the `lsing_sw` repository on branch `feat/monotonic-kan-transport`.
The project implements **Monotonic Kolmogorov-Arnold Network (Monotonic KAN) Transport Maps** for **L-SING** (*Localized Sparsity Identification for Non-Gaussian Distributions*, accepted at AAAI'25, Liaw et al.).

In L-SING, local 1D conditional transport maps $S^k(x_{-k}, \cdot)$ push forward conditional distributions $\pi(x_k \mid x_{-k})$ to a standard normal reference $\eta = \mathcal{N}(0, 1)$. We replaced the baseline UMNN (Unconstrained Monotonic Neural Networks) with Monotonic KANs and Scale-Modulated KANs (SM-KANs) to eliminate numerical quadrature (50-node Clenshaw-Curtis grids) and enable exact, closed-form analytic derivatives for generalized precision matrix extraction ($\hat{\Omega}_{jk}$).

---

## 2. Current Architecture (`models/monotonic_kan/`)
The Monotonic KAN codebase has been modularized into its own standalone sub-package under `models/monotonic_kan/`, mirroring the baseline `models/UMNN/` sub-package:

```
models/
├── __init__.py               # Re-exports top-level classes from UMNN and monotonic_kan
├── setup.py                  # Configured with packages=['UMNN', 'monotonic_kan']
├── monotonic_kan/            # Monotonic KAN sub-package
│   ├── __init__.py           # Exports BSplineBasis, KANLayer, DeepKAN, MonotonicKANMap, ScaleModulatedKANMap
│   ├── basis.py              # Uniform cubic B-spline knot grid & vectorized Cox-de Boor engine
│   ├── layer.py              # KANLayer (per-edge monotonic masks) & DeepKAN (multi-layer unconstrained KAN)
│   └── kan_map.py            # Unified MonotonicKANMap & ScaleModulatedKANMap (arbitrary depth & closed-form derivatives)
└── UMNN/                     # Original baseline sub-package
```

### Key Implementation Principles:
1. **Unified Arbitrary Depth ($L \ge 2$):**
   - No 2-layer special-case branching (`if self.L == 2:` removed everywhere; 2-layer is simply an $L=2$ instance of the general architecture).
   - `core_layers = nn.ModuleList([KANLayer(...) for ...])`:
     - **Layer 0 ($l=0$):** Only target feature $x_k$ is monotonic (`mask[:, target_index] = True`), conditioning features are unconstrained.
     - **Downstream layers ($l \ge 1$):** All edges are strictly monotonic (`mask[:, :] = True`).
   - Scale ($\sigma$) and offset ($\alpha$) networks are instances of `DeepKAN`:
     - Single-layer additive is simply `DeepKAN([d_cond, 1])`.
     - Multi-layer scale is `DeepKAN([d_cond, m_1, ..., 1])`.

2. **Programmatic Einstein Summation Recurrence:**
   - Evaluates 1st, 2nd, and 3rd-order derivatives ($S^k, \partial_k S^k, \partial_j S^k, \partial_j \partial_k S^k, \partial_k^2 S^k, \partial_j \partial_k^2 S^k$) in closed form via layer-by-layer `torch.einsum` recurrence across $(B, n_l, d)$ tensors without autograd.

3. **Cumulative Softplus & Active Backbone Path Tracking:**
   - Strictly enforces $c_{m+1} - c_m = \operatorname{softplus}(\theta_m) + \delta_{\text{mono}} > \delta_{\text{mono}} > 0$ with $C^1$ linear tail extrapolation as required by Section 6.1 of `theory/kan_transport.tex`.
   - Tracks the certified active backbone path $\mathcal{P}^* = (i_0^*, i_1^*, \dots, i_L^*)$ via `model.active_path` and `derivs["S_k_backbone"]`, verifying $\partial_k S^k(x) \ge D_{\mathcal{P}^*}(x) > 0$.

4. **Default Regularization:**
   - `penalty_type="functional"` is the default across all models and trainers (Equation 157 in `theory/kan_transport.tex` / Equation 5 in Liaw et al.).

---

## 3. Cleanups Completed
1. **Redundancies Removed:**
   - Deleted the obsolete `models/kan_transport.py` shim.
   - Completely purged all unused `CubicHermiteBasis` code and unit tests.
   - Removed all `self.L == 2` branches, separate `outer_delta`, and duplicate scale net vs weights branches.
2. **Benchmarks Updated:**
   - Renamed `benchmarks/run_ablation.py` $\to$ `benchmarks/run_sparsity_ablation.py`.
   - `benchmarks/run_gaussian.py`, `run_butterfly_40d.py`, `run_butterfly_samples.py` standardized to compare with the functional penalty.
   - `benchmarks/run_scale_modulated_butterfly.py` stripped of Additive Monotonic KAN, focusing solely on Scale-Modulated KAN.

---

## 4. Verification & Testing
All 22 unit tests pass:
```bash
PYTHONPATH=.:src .venv/bin/pytest tests/
# 22 passed, 1 warning in ~5.2s
```
- Includes tests verifying arbitrary depth (3-layer and 4-layer KANs) against float64 PyTorch autograd ground truth.
- Verified that all benchmark scripts compile and run cleanly.

---

## 5. Next Steps / Action Items
1. Review the unified `models/monotonic_kan/kan_map.py` to ensure it adheres to the desired team coding style.
2. Stage and commit the cleaned modular repository changes (`git status`).
3. Prepare repository documentation/README updates before sharing with collaborators.
