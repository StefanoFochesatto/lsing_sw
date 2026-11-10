# Action Plan: Monotonic KAN Transport Maps Paper (`theory/kan_transport.tex`)
**Selected Framing:** **Option A / Path 2 — Local B-Splines vs. Global Spectral SOS-Chebyshev Duality + $C^3$ Universal Approximation Theory**

---

## 1. Strategic Overview & Narrative Arc

Rather than cluttering the manuscript with a four-basis "basis zoo" (cutting Bernstein as redundant with B-splines and cutting Integrated RBF due to its curvature ripples on linear/Gaussian edges), the manuscript presents a clean, classical numerical-analysis duality within our certified **Monotonic KAN / Scale-Modulated KAN (SM-KAN)** framework:

1. **Local Piecewise Polynomial Basis (Uniform B-Splines — Section 4.1):**
   - Provides the geometric foundation of KANs (Liu et al., 2024) with localized compact support, certifying $\phi'(t) \ge \epsilon > 0$ via discrete control-point increments $c_{m+1} - c_m \ge \delta > 0$ (Lemma 4.1).
   - Serves as the pedagogical baseline for all existing architecture diagrams (`Figures 1–6`) and structural theorems (**Theorems 4.3, 4.4, Proposition 4.5, and the Hessian Cancellation Theorem 6.1**, all of which are 100% basis-agnostic).
2. **Global Orthogonal Spectral Basis (Sum-of-Squares Chebyshev — Section 5.4):**
   - Motivates moving from a local knot-grid spline to a **global spectral polynomial basis** on $[a, b]$ for three concrete reasons:
     1. **Eliminating Knot-Boundary Discontinuities ($C^\infty$ Smoothness):** Cubic B-splines ($p=3$) are only $C^2$ across interior knots, producing step discontinuities in the third derivative $\phi'''(t)$ required by the generalized precision integrand $T_{jk}(x)$. SOS-Chebyshev edge functions are $C^\infty$ across the entire active domain $(a, b)$.
     2. **Eliminating Cox–de Boor Knot-Lookup Overhead:** B-spline evaluation requires piecewise knot-span indexing and recursive division, causing memory-bound branch divergence. SOS-Chebyshev evaluates globally via a branch-free 3-term recurrence and exact closed-form antiderivatives $\Psi_n(u) = \int_0^u T_n(s)\,ds$, training **$1.7\times$ faster** than B-splines.
     3. **Spectral Convergence & Exact Linear Identity:** By the Markov–Lukács representation theorem, parameterizing $\phi'(u) = \mathbf{T}_K(u)^\top (L L^\top) \mathbf{T}_K(u) + \delta \ge \delta > 0$ via a lower-triangular Cholesky factor $L$ guarantees global strict monotonicity, represents pure linear functions identically (when higher rows of $L$ vanish), and achieves the **lowest Test NLL (`0.1563`) and lowest precision Frobenius error** across all benchmarks.
3. **Constructive $C^3$ Universal Approximation & Precision Consistency (Section 4.5):**
   - Proves that both the local B-spline basis (via variation-diminishing Schoenberg spline approximation as mesh size $h \to 0$) and the global SOS-Chebyshev basis (via rank-1 square-root spectral Chebyshev approximation as degree $K \to \infty$) are **universal approximators of orientation-preserving diffeomorphisms in the $C^3([a, b])$ norm**.
   - Extends this to **heteroscedastic location–scale** and **multi-branch conditional transport maps** in $C^3(K)$, and proves (**Corollary 4.8**) that $C^3(K)$ convergence guarantees uniform convergence of the generalized precision integrand $T_{jk}(x)$ and graph recovery matrix $\Omega_{jk}$.

---

## 2. Branch & Codebase Status

| Branch | Status | Contents & Role |
| :--- | :--- | :--- |
| **`main`** & **`feat/monotonic-kan-transport`** | **Merged & Active** | Complete Monotonic KAN & SM-KAN stack (`models/monotonic_kan/`) with B-Spline (`basis_type="bspline"`) and SOS-Chebyshev (`basis_type="chebyshev"`), `theory/kan_transport.tex`, custom figures (`theory/figures/`), benchmark suites (`benchmarks/run_all_bases_benchmark.py`, `results/kan_benchmarks/all_bases_benchmark_results.json`), 39/39 passing unit tests, and this Action Plan. |
| **`feat/sm-mlp-transport`** | Saved (`e6df3e5`) | **Saved Companion Paper Branch:** Contains all SM-MLP (`models/sm_mlp_transport.py`), DSM-MLP (`models/dsm_mlp_transport.py`), `docs/DSM_MLP_APPROXIMATION_THEORY.md`, `docs/SM_MLP_VS_KAN_THEORETICAL_SYNTHESIS.md`, and the 8-model NVIDIA A100 benchmark suites (`results/kan_benchmarks/A100_*`). |

---

## 3. Surgical Section-by-Section Edit Plan for `theory/kan_transport.tex`

### Phase 1: Abstract & Section 1 Introduction Polish (`lines 52–76`)

1. **Complete the Abstract (`lines 52–54`):**
   - Replace the trailing `"In this work…"` with a complete summary covering:
     1. The four formal admissibility criteria for L-SING conditional transport maps (strict target positivity $\partial_k S^k \ge \epsilon > 0$, global $\mathbb{R}^d$ domain extrapolation, $C^3$ smoothness for third-order mixed log-density derivatives, and unconstrained conditioning freedom).
     2. Two-Layer, Deep, and **Scale-Modulated Monotonic KANs (SM-KAN)** that resolve the algebraic polarization obstruction of purely additive monotonic networks on multiplicative/heteroscedastic couplings.
     3. Both **local piecewise polynomial (B-spline)** and **global spectral Sum-of-Squares Chebyshev (SOS-Chebyshev)** monotonic edge bases, together with constructive **$C^3$ Universal Approximation theorems** and the **Hessian Cancellation Theorem** for analytic precision matrix extraction.
     4. Empirical validation demonstrating $84\times$ to $300\times$ faster precision matrix extraction and $284\times$ lower memory consumption than numerical-quadrature UMNNs across Gaussian, multiplicative Butterfly, mixed-coupling, and 156-gene TCGA Ovarian Cancer networks.
2. **Uncomment Section 1 (`lines 60–75`):**
   - Uncomment `\section{Introduction and Problem Setting}` so the manuscript opens cleanly before the L-SING measure transport formulation.

---

### Phase 2: Global Spectral Basis — Sum-of-Squares (SOS) Chebyshev (`Insert as Section 5.4, after line 668`)

After Section 5.1–5.3 establishes the local B-spline implementation (cumulative softplus increments, $C^1$ linear tail extrapolation, and Cox–de Boor derivative recurrence), add **Section 5.4: Global Spectral Parameterization via Sum-of-Squares Chebyshev Polynomials**:

1. **Motivation (Local Splines vs. Global Spectral Polynomials):**
   - Explain the two practical trade-offs of local cubic B-splines:
     - Across interior knots, cubic B-splines ($p=3$) are only $C^2$, so $\phi'''(t)$ exhibits piecewise constant jumps at knot boundaries unless $p \ge 4$.
     - Evaluating B-splines requires piecewise knot-interval lookup and recursive Cox–de Boor sweeps across knot spans.
   - Introduce the **SOS-Chebyshev basis** on the affine-normalized coordinate $u(t) = \frac{2t - (a+b)}{b-a} \in [-1, 1]$ for $t \in [a, b]$ as a knot-free, $C^\infty$ global spectral alternative.

2. **Lemma & Proof (Monotonicity and Exact Antiderivative of SOS-Chebyshev):**

```latex
\begin{lemma}[Monotonicity and Closed-Form Integration of Sum-of-Squares Chebyshev Polynomials]
\label{lem:chebyshev_mono}
Let $u(t) = \frac{2t - (a + b)}{b - a} \in [-1, 1]$ on $[a, b]$ and let $\mathbf{T}_K(u) = \big(T_0(u), T_1(u), \dots, T_K(u)\big)^\top$ denote the vector of first-kind Chebyshev polynomials up to degree $K \ge 1$. For any unconstrained lower-triangular factor $L \in \mathbb{R}^{(K+1) \times (K+1)}$ and strict positivity floor $\delta_{\text{mono}} > 0$, define the positive semi-definite matrix $Q = L L^\top \succeq 0$ and parameterize the derivative with respect to $u$ on $[-1, 1]$ as the Sum-of-Squares quadratic form,
\begin{equation}
    \frac{d\phi}{du}(u) = \mathbf{T}_K(u)^\top \big( L L^\top \big) \mathbf{T}_K(u) + \delta_{\text{mono}} = (c_0 + \delta_{\text{mono}}) T_0(u) + \sum_{n=1}^{2K} c_n T_n(u) \ge \delta_{\text{mono}} > 0,
    \label{eq:chebyshev_sos_deriv}
\end{equation}
where the expanded Chebyshev coefficients $c_n$ ($0 \le n \le 2K$) are given by the product-to-sum contraction,
\begin{equation}
    \mathbf{T}_K(u)^\top Q \mathbf{T}_K(u) = \sum_{j=0}^K \sum_{r=0}^K Q_{jr} \frac{1}{2}\Big( T_{j+r}(u) + T_{|j-r|}(u) \Big) = \sum_{n=0}^{2K} c_n T_n(u).
    \label{eq:chebyshev_contraction}
\end{equation}
Then:
\begin{enumerate}
    \item \textbf{Closed-Form Activation:} Integrating \eqref{eq:chebyshev_sos_deriv} with respect to $t$ yields the exact polynomial activation on $[a, b]$,
    \begin{equation}
        \phi(t) = c_{\text{bias}} + \frac{b - a}{2} \left[ (c_0 + \delta_{\text{mono}}) \Psi_0\big(u(t)\big) + \sum_{n=1}^{2K} c_n \Psi_n\big(u(t)\big) \right],
        \label{eq:chebyshev_antideriv}
    \end{equation}
    where $\Psi_n(u) \coloneqq \int_0^u T_n(s)\,ds$ is given in closed form by $\Psi_0(u) = u$, $\Psi_1(u) = \frac{1}{4}(T_2(u) - 1)$, and
    \begin{equation}
        \Psi_n(u) = \frac{1}{2}\left( \frac{T_{n+1}(u)}{n+1} - \frac{T_{n-1}(u)}{n-1} \right) - \frac{1}{2}\left( \frac{T_{n+1}(0)}{n+1} - \frac{T_{n-1}(0)}{n-1} \right), \quad n \ge 2.
    \end{equation}
    \item \textbf{Global Monotonicity and $C^\infty$ Interior Regularity:} On $[a, b]$, $\phi \in C^\infty([a, b])$ and its derivative with respect to $t$ satisfies the uniform strictly positive bound,
    \begin{equation}
        \phi'(t) = \frac{2}{b - a} \frac{d\phi}{du}\big(u(t)\big) \ge \frac{2\,\delta_{\text{mono}}}{b - a} \eqqcolon \epsilon > 0 \quad \forall t \in [a, b],
    \end{equation}
    with closed-form higher derivatives $\phi''(t) = \big(\frac{2}{b-a}\big)^2 \sum_{n=1}^{2K} c_n T'_n(u(t))$ and $\phi'''(t) = \big(\frac{2}{b-a}\big)^3 \sum_{n=2}^{2K} c_n T''_n(u(t))$.
    \item \textbf{Global $C^1$ Tail Extrapolation:} Extending $\phi(t)$ outside $[a, b]$ via the linear tail formula \eqref{eq:linear_extrap} with exact boundary slopes $\phi'(a) = \frac{2}{b-a}\big(\delta_{\text{mono}} + \sum_{n=0}^{2K} (-1)^n c_n\big) \ge \epsilon$ and $\phi'(b) = \frac{2}{b-a}\big(\delta_{\text{mono}} + \sum_{n=0}^{2K} c_n\big) \ge \epsilon$ preserves $\phi'(t) \ge \epsilon > 0$ globally across all $t \in \mathbb{R}$.
\end{enumerate}
\end{lemma}
```

---

### Phase 3: $C^3$ Universal Approximation Theory & Precision Consistency (`Insert as Section 4.5, after line 597, or Section 5.5`)

Add a dedicated theoretical subsection proving constructive **$C^3$ Universal Approximation** for both the local B-spline and global SOS-Chebyshev parameterizations, and connecting $C^3$ convergence directly to L-SING graph recovery.

#### 1. Theorem: Constructive Univariate $C^3([a, b])$ Monotone Universal Approximation
```latex
\begin{theorem}[Univariate $C^3$ Universal Approximation of Monotonic Spline and Spectral Edges]
\label{thm:univariate_c3_universality}
Let $[a, b] \subset \mathbb{R}$ be a compact interval and let $f \in C^3([a, b])$ be any orientation-preserving target diffeomorphism satisfying $\inf_{t \in [a, b]} f'(t) \ge \eta_0 > 0$. Then for every $\varepsilon > 0$:
\begin{enumerate}
    \item \textbf{Global SOS-Chebyshev Density:} There exist a degree $K \ge 1$, a lower-triangular factor $L \in \mathbb{R}^{(K+1)\times(K+1)}$ (which may be chosen of rank 1), a floor $\delta_{\text{mono}} \in (0, \frac{b-a}{4}\eta_0)$, and an intercept $c_{\text{bias}} \in \mathbb{R}$ such that the SOS-Chebyshev activation $\phi_K$ of Lemma~\ref{lem:chebyshev_mono} satisfies $\inf_{t \in \mathbb{R}} \phi'_K(t) \ge \eta_0 / 2 > 0$ and
    \begin{equation}
        \| \phi_K - f \|_{C^3([a, b])} \coloneqq \max_{0 \le r \le 3} \sup_{t \in [a, b]} \big| \phi_K^{(r)}(t) - f^{(r)}(t) \big| < \varepsilon.
        \label{eq:c3_norm_bound}
    \end{equation}
    \item \textbf{Local B-Spline Density:} For any fixed spline degree $p \ge 4$, there exist a sufficiently fine knot sequence $a = t_{p+1} < \dots < t_{M+1} = b$ and strictly increasing control points $c_{m+1} - c_m \ge \delta > 0$ (parameterized via \eqref{eq:softplus_reparam}) such that the B-spline activation $\phi_M$ of Lemma~\ref{lem:bspline_mono} satisfies $\inf_{t \in \mathbb{R}} \phi'_M(t) \ge \eta_0 / 2 > 0$ and $\| \phi_M - f \|_{C^3([a, b])} < \varepsilon$.
\end{enumerate}
\end{theorem}
```

#### Complete Algebraic Proof (Fluid Narrative):
- **Part 1 (SOS-Chebyshev via Rank-1 Square-Root Spectral Approximation):**
  - Rescale to $u \in [-1, 1]$ via $t(u) = \frac{b-a}{2}u + \frac{a+b}{2}$ and set $p(u) \coloneqq \frac{b-a}{2} f'(t(u)) \ge \frac{b-a}{2}\eta_0 > 0$. Fix $\delta_{\text{mono}} = \frac{b-a}{4}\eta_0 > 0$, so that $p(u) - \delta_{\text{mono}} \ge \delta_{\text{mono}} > 0$ on $[-1, 1]$.
  - Because $f \in C^3([a, b])$, we have $p \in C^2([-1, 1])$, and since $z \mapsto \sqrt{z}$ is $C^\infty$ on $[\delta_{\text{mono}}, \infty)$, the square-root function $g(u) \coloneqq \sqrt{p(u) - \delta_{\text{mono}}}$ belongs to $C^2([-1, 1])$.
  - By classical Jackson/Weierstrass simultaneous polynomial approximation in $C^2([-1, 1])$, for any $\tilde{\varepsilon} > 0$ there exists a degree-$K$ Chebyshev expansion $q_K(u) = \sum_{k=0}^K a_k T_k(u) = \mathbf{a}^\top \mathbf{T}_K(u)$ such that $\max_{0 \le r \le 2} \sup_{u \in [-1, 1]} |q_K^{(r)}(u) - g^{(r)}(u)| < \tilde{\varepsilon}$.
  - Selecting the rank-1 lower-triangular Cholesky factor $L = [\mathbf{a}, \mathbf{0}, \dots, \mathbf{0}] \in \mathbb{R}^{(K+1)\times(K+1)}$ gives $L L^\top = \mathbf{a}\mathbf{a}^\top \succeq 0$ and
    $$\frac{d\phi_K}{du}(u) = \mathbf{T}_K(u)^\top (L L^\top) \mathbf{T}_K(u) + \delta_{\text{mono}} = q_K(u)^2 + \delta_{\text{mono}}.$$
  - Since $(q_K^2)' = 2 q_K q'_K$ and $(q_K^2)'' = 2 (q'_K)^2 + 2 q_K q''_K$, uniform convergence of $q_K, q'_K, q''_K$ to $g, g', g''$ on the compact interval $[-1, 1]$ implies uniform convergence of $\frac{d\phi_K}{du}$ and its first two derivatives to $g^2 + \delta_{\text{mono}} = p$ and $p', p''$. Integrating from $a$ to $t$ with $\phi_K(a) = f(a)$ yields $\|\phi_K - f\|_{C^3([a, b])} < \varepsilon$.
- **Part 2 (B-Splines via de Boor / Schoenberg Derivative Approximation):**
  - By Lemma 4.1, the derivative of a degree-$p$ B-spline ($p \ge 4$) is a degree-$(p-1)$ B-spline $\phi'_M(t) = \sum_{m=2}^M d_m B_{m, p-1}(t)$ with coefficients $d_m = p \frac{c_m - c_{m-1}}{t_{m+p}-t_m}$.
  - Since $f' \in C^2([a, b])$ and $p - 1 \ge 3$, standard spline quasi-interpolation / Schoenberg variation-diminishing approximation of $f'$ on a uniform mesh of width $h = (b-a)/(M-p) \to 0$ constructs positive derivative coefficients $d_m \ge \eta_0 / 2 > 0$ such that $\|\phi'_M - f'\|_{C^2([a, b])} \to 0$ as $h \to 0$.
  - Setting increments $c_m - c_{m-1} = \frac{t_{m+p}-t_m}{p} d_m > 0$ and anchoring $c_1 = f(a)$ integrates $\phi'_M$ exactly, yielding $\|\phi_M - f\|_{C^3([a, b])} < \varepsilon$.

#### 2. Proposition & Corollary: Conditional Location–Scale Universality and Consistency of $\Omega_{jk}$
- **Proposition (Location–Scale & Multi-Branch Conditional $C^3(K)$ Approximation):**
  - Establish that SM-KAN (Definition 4.3) universally approximates any heteroscedastic location–scale conditional transport map $S^{k,*}(x) = E(\sigma^*(x_{-k})) f^*(x_k) + \alpha^*(x_{-k})$ in $C^3(K)$ on compact sets $K = K_{-k} \times [a, b] \subset \mathbb{R}^d$, and that a $Q$-branch scale-modulated expansion $S^k(x) = \sum_{q=1}^Q E_q(x_{-k}) \phi_{q, k}(x_k) + \alpha(x_{-k})$ (where $E_q(x_{-k}) > 0$ are universal positive conditioning networks) achieves full joint multivariate $C^3(K)$ density over all $C^3$ conditional transport maps with $\partial_k S^{k,*}(x) > 0$.
- **Corollary (`cor:precision_c3_consistency` — Consistency of Generalized Precision Discovery):**
  - Show that if $\|S^k_N - S^{k,*}\|_{C^3(K)} \to 0$ on a compact support $K$ where $\partial_k S^{k,*}(x) \ge \eta_0 > 0$, then because the generalized precision integrand $T_{jk}(x) = -S_j S_k - S S_{jk} + (S_{jkk}S_k - S_{kk}S_{jk})/S_k^2$ in Eq. (683) has denominator $S_k^2 \ge \eta_0^2 / 4 > 0$ bounded uniformly away from zero for large $N$, we have uniform convergence $\sup_{x \in K} |T_{jk}(x; S^k_N) - \partial_j \partial_k \log \pi(x_k \mid x_{-k})| \to 0$ and $\Omega_{jk}(S^k_N) \to \Omega_{jk}(S^{k,*})$. (Note explicitly that mere $C^0$ universal approximation fails to control $T_{jk}$.)

---

### Phase 4: Empirical Evaluation Updates (`Section 7, lines 770–820`)

Add **Subsection 7.2: Local Spline vs. Global Spectral Basis Comparison (B-Spline vs. SOS-Chebyshev)** with a clean 2-way comparison table drawn from `results/kan_benchmarks/all_bases_benchmark_results.json` and `A100_FULL_8MODEL_BENCHMARK_REPORT.md`:

| Metric / Property | SM-KAN (Local B-Spline) | SM-KAN (Global SOS-Chebyshev) |
| :--- | :---: | :---: |
| **Basis Formulation** | Piecewise degree-$p$ Cox–de Boor spline | Global degree-$2K$ Markov–Lukács SOS polynomial |
| **Interior Smoothness** | $C^2$ at knots ($p=3$) or $C^{p-1}$ ($p \ge 4$) | **$C^\infty$ globally on $(a, b)$** |
| **Monotonicity Mechanism** | Cumulative softplus increments $c_{m+1} - c_m > 0$ | Cholesky quadratic form $\mathbf{T}_K^\top (LL^\top) \mathbf{T}_K + \delta > 0$ |
| **Derivative Recurrence ($B=1024$)** | $19.20\text{ ms}$ | **$11.07\text{ ms}$ ($1.73\times$ faster)** |
| **6D Mixed Coupling Train Time** | $14.89\text{ s}$ | **$8.69\text{ s}$ ($1.71\times$ faster)** |
| **6D Mixed Coupling Test NLL** | $0.1786$ ($0.440$ on A100) | **$0.1563$ ($0.392$ on A100 — Best)** |
| **10D Gaussian Recovery ($F_1$ / Fro. Err.)** | $F_1 = 1.000$ / $2.4816$ | **$F_1 = 1.000$ / $2.0271$ (Best)** |
| **Max Error vs. Float64 Autograd ($\partial_j\partial_k^2 S^k$)** | $5.77 \times 10^{-15}$ | **$4.44 \times 10^{-16}$ (Machine $\epsilon$)** |

---

## 4. Execution Checklist

1. [x] Merge `feat/monotonic-kan-transport` and `feat/integrated-rbf-chebyshev` into `main` and verify all 39/39 unit tests pass.
2. [ ] **Phase 1:** Complete the Abstract and uncomment Section 1 in `theory/kan_transport.tex`.
3. [ ] **Phase 2:** Add Section 5.4 (*Global Spectral Parameterization via Sum-of-Squares Chebyshev Polynomials*) with Lemma `\ref{lem:chebyshev_mono}` and its proof.
4. [ ] **Phase 3:** Add Section 4.5 (*Universal Approximation in the $C^3$ Norm and Precision Matrix Consistency*) with Theorem `\ref{thm:univariate_c3_universality}` (proving both SOS-Chebyshev and B-spline $C^3$ universality), the conditional location–scale / multi-branch proposition, and Corollary `\ref{cor:precision_c3_consistency}`.
5. [ ] **Phase 4:** Add Section 7.2 comparing Local B-Splines vs. Global SOS-Chebyshev on speed, numerical precision, NLL, and graph recovery.
6. [ ] Compile `theory/kan_transport.tex` via `latexmk -pdf -interaction=nonstopmode -cd theory/kan_transport.tex` and verify zero LaTeX errors or undefined references.
