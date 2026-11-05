# Scale-Modulated Monotonic KANs (SM-KAN) for Autoregressive Density Estimation & Generative Modeling

## Executive Summary
This document outlines the roadmap and technical specifications for expanding **Scale-Modulated Monotonic Kolmogorov–Arnold Networks (SM-KAN)** from localized precision matrix estimation into a **full-fledged autoregressive density estimator and generative normalizing flow**.

All experiments and code for this effort reside on branch:
```bash
git checkout feat/smkan-density-estimation
```

---

## 1. Mathematical Formulation

Let $X = (X_1, \dots, X_d) \sim \pi$ denote a continuous random vector on $\mathbb{R}^d$. A lower-triangular Knothe–Rosenblatt (KR) rearrangement parameterizes the transport map $S: \mathbb{R}^d \to \mathbb{R}^d$ as an autoregressive triangular system:
$$
S(x) = \begin{pmatrix}
S^1(x_1) \\
S^2(x_1, x_2) \\
\vdots \\
S^d(x_1, \dots, x_d)
\end{pmatrix},
$$
where each component map $S^k(x_{<k}, x_k): \mathbb{R}^k \to \mathbb{R}$ pushes forward the conditional measure $\pi(x_k \mid x_{<k})$ to a 1D standard Gaussian reference $\eta_k = \mathcal{N}(0, 1)$.

### The Conditional SM-KAN Parameterization
Each coordinate map $S^k$ is parameterized via Scale-Modulated Monotonic KAN:
$$
S^k(x_{<k}, x_k) = E\big(\sigma(x_{<k})\big) \cdot S^k_{\text{core}}(x_{<k}, x_k) + \alpha(x_{<k}),
$$
where:
1. **Certified Monotonic Core ($S^k_{\text{core}}$):** An $L$-layer KAN ($L \ge 2$, typically $L=2$) satisfying target monotonicity $\partial_k S^k_{\text{core}} \ge \epsilon > 0$ via cumulative softplus control points and $C^1$ linear tail extrapolation.
2. **Conditioning Scale Network ($\sigma$):** An unconstrained B-spline network $\sigma(x_{<k}) = \sum_{j < k} \psi_j(x_j)$.
3. **Conditioning Offset Network ($\alpha$):** An unconstrained B-spline network $\alpha(x_{<k}) = \sum_{j < k} \rho_j(x_j)$.
4. **Smooth Saturated Scale Activation ($E$):**
   $$
   E(\sigma) \coloneqq \exp\left( s_{\max} \tanh\left( \frac{\sigma}{s_{\max}} \right) \right) \in \big(e^{-s_{\max}}, e^{s_{\max}}\big), \quad s_{\max} = 4.0.
   $$

### Closed-Form Joint Density (Likelihood Evaluation)
Because $\nabla S(x)$ is lower-triangular, the Jacobian determinant is the product of diagonal target derivatives:
$$
\det \nabla S(x) = \prod_{k=1}^d \partial_k S^k(x_{<k}, x_k) > 0.
$$
The exact log-density of the model under the change-of-variables formula is computed analytically without numerical integration or matrix determinants:
$$
\log \pi_S(x) = -\frac{d}{2}\log(2\pi) - \frac{1}{2}\sum_{k=1}^d \big(S^k(x_{<k}, x_k)\big)^2 + \sum_{k=1}^d \log \partial_k S^k(x_{<k}, x_k).
$$

### Exact Generative Inversion (Sampling $x \sim \pi_S$)
To generate samples from the learned distribution:
1. Sample $z \sim \mathcal{N}(0, I_d)$.
2. Solve $S(x) = z$ coordinate-by-coordinate for $k = 1, \dots, d$:
   $$
   S^k(x_{<k}, x_k) = z_k \iff S^k_{\text{core}}(x_{<k}, x_k) = \frac{z_k - \alpha(x_{<k})}{E(\sigma(x_{<k}))} \eqqcolon y_k.
   $$
3. Because $x_k \mapsto S^k_{\text{core}}(x_{<k}, x_k)$ is strictly increasing with exact analytic derivatives $\partial_k S^k_{\text{core}}$ and $\partial_k^2 S^k_{\text{core}}$, 1D Newton–Raphson root finding converges with quadratic rates in **3 to 5 iterations**:
   $$
   x_k^{(t+1)} = x_k^{(t)} - \frac{S^k(x_{<k}, x_k^{(t)}) - z_k}{\partial_k S^k(x_{<k}, x_k^{(t)})}.
   $$

---

## 2. Competitive Landscape: Comparison Against Existing Density Estimators

| Architecture | Paradigm | Forward Likelihood | Log-Det Jacobian | Generative Inversion | Key Bottlenecks Overcome by SM-KAN |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RealNVP** (Dinh et al., 2017) | Affine coupling | Analytic (Fast) | Analytic (Diagonal) | Exact (One-pass) | Constrained to affine conditional responses; cannot model S-curves or inflections. |
| **MAF** (Papamakarios et al., 2017) | Affine autoregressive | Analytic (Fast) | Analytic (Diagonal) | Sequential ($O(d)$) | Affine assumption restricts expressivity per layer; needs deep stacking. |
| **FFJORD** (Grathwohl et al., 2018) | Continuous flow (ODE) | Numerical ODE solve | Hutchinson trace estimator | Numerical ODE solve | Requires heavy numerical ODE integration for training and inference; slow convergence. |
| **UMNN-MAF** (Wehenkel & Louppe, 2019) | Monotonic integral flow | Numerical quadrature (50 nodes) | Exact (Quadrature integrand) | Numerical root-finding | Heavy computational quadrature overhead; backpropagation through numerical integrators. |
| **RQ-NSF** (Durkan et al., 2019) | Rational-quadratic splines | Analytic (Spline bins) | Analytic (Spline slopes) | Exact analytical inversion | Splines defined only on $[-B, B]$; requires bin width/height parameterization by MLPs. |
| **SM-KAN (Ours)** | Scale-Modulated Monotonic KAN | **Exact closed-form** | **Exact closed-form** | **Fast 1D Newton (3-5 iters)** | **Zero numerical quadrature; global $C^1$ extrapolation on $\mathbb{R}^d$; native B-spline smoothness.** |

---

## 3. Standard Online Benchmarks & Published SOTA Results

### Benchmark 1: Five Standard Real-World Tabular Datasets (Papamakarios et al., 2017)
Evaluated by computing the **average test log-likelihood** (in *nats*, higher is better):

| Dataset | Dimensions ($d$) | Train Size | Description |
| :--- | :---: | :---: | :--- |
| **POWER** | 6 | $\approx 1{,}650{,}000$ | Electric power consumption |
| **GAS** | 8 | $\approx 850{,}000$ | Gas sensor array drift |
| **HEPMASS** | 21 | $\approx 315{,}000$ | High-energy physics collision particles |
| **MINIBOONE** | 43 | $\approx 29{,}500$ | Fermilab neutrino beam vs. background |
| **BSDS300** | 63 | $\approx 1{,}000{,}000$ | Natural image patches ($8 \times 8$) |

#### Published Benchmark Log-Likelihoods (nats)
| Model | POWER ($6d$) | GAS ($8d$) | HEPMASS ($21d$) | MINIBOONE ($43d$) | BSDS300 ($63d$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RealNVP** | $0.17$ | $8.33$ | $-18.71$ | $-13.55$ | $153.28$ |
| **MAF** | $0.24$ | $10.08$ | $-17.70$ | $-11.75$ | $155.69$ |
| **Glow** | $0.42$ | $12.24$ | $-16.99$ | $-10.55$ | $156.95$ |
| **FFJORD** | $0.46$ | $8.59$ | $-14.92$ | $-10.43$ | $157.40$ |
| **UMNN-MAF** | $0.63$ | $10.89$ | $-13.99$ | $-9.67$ | $157.98$ |
| **RQ-NSF (C)** | $0.64$ | $12.80$ | $-14.27$ | $-8.72$ | $157.36$ |
| **RQ-NSF (AR)** | $0.66$ | $13.09$ | $-14.01$ | $-9.22$ | $157.31$ |
| **TraDE** | **0.73** | **13.27** | **-12.01** | **-9.49** | **160.01** |

---

### Benchmark 2: Canonical 2D Density Estimation Testbeds
Used for qualitative visual evaluation of mode recovery, sharp density boundaries, and sampling fidelity:
1. **Two Moons:** Two interlocking non-convex crescent densities.
2. **8 Gaussians (Ring):** Circular mixture of 8 isolated Gaussian modes with low-density basins.
3. **Pinwheel:** Multi-spiral distributions with sharp non-linear curvature.
4. **Swiss Roll:** 2D non-linear manifold winding.
5. **Checkerboard:** Discontinuous density grid with sharp piecewise boundaries.

---

## 4. Implementation Plan on `feat/smkan-density-estimation`

### Phase 1: Autoregressive Density Estimator Module
- File: `models/sm_kan_density.py`
- Class: `SMKANDensityEstimator(torch.nn.Module)`
- Composes $d$ component maps $S^k(x_{<k}, x_k)$ using `MonotonicKANMap` with `conditioning_scale=True`.
- Methods:
  - `log_prob(x) -> torch.Tensor` of shape `(B,)` computing the exact log pullback density.
  - `forward(x) -> z, log_det_jacobian` mapping samples to latent Gaussian space.
  - `sample(n_samples) -> x` sampling $z \sim \mathcal{N}(0, I_d)$ and inverting fiber-by-fiber.

### Phase 2: Vectorized Root-Finding Inversion Engine
- File: `models/root_finding.py`
- Implements batched 1D Newton–Raphson with fallback to bisection for guaranteed global convergence outside active knot spans.
- Benchmarks convergence rate (iterations vs. residual tolerance $\|S^k(x) - z_k\| < 10^{-6}$).

### Phase 3: 2D Synthetic Benchmark Suite
- Script: `benchmarks/run_2d_density_benchmark.py`
- Compares SM-KAN against RealNVP and UMNN on Two Moons, 8 Gaussians, and Pinwheel.
- Generates side-by-side density contour plots and generated sample scatter plots.

### Phase 4: UCI Tabular Benchmark Suite
- Script: `benchmarks/run_tabular_density_benchmark.py`
- Evaluates average test log-likelihood on POWER and GAS.
- Records training wall-clock time, parameter count, and sampling speed vs. published numbers.
