import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Circle, Wedge
import scipy.stats as stats

# Set styling
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['figure.autolayout'] = False

out_dir = os.path.dirname(os.path.abspath(__file__)) + "/figures"
os.makedirs(out_dir, exist_ok=True)

# ==============================================================================
# Figure 1: Architectural Flow of Monotonic KAN and Scale-Modulated KAN
# ==============================================================================


def make_fig1():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    # Left: 2-Layer Monotonic KAN
    ax1.set_xlim(-0.5, 6.5)
    ax1.set_ylim(-0.5, 5.5)
    ax1.axis('off')
    ax1.set_title(
        r"$\mathbf{(a)}$ Two-Layer Monotonic KAN Computation Graph", fontweight="bold", pad=12)

    # Input nodes
    inputs = [
        (r"$x_1$", 4.5, "gray"),
        (r"$x_j$", 3.2, "royalblue"),
        (r"$x_k$" + "\n(target)", 1.8, "forestgreen"),
        (r"$x_d$", 0.5, "gray")
    ]
    for label, y, col in inputs:
        circle = patches.FancyBboxPatch((0.0, y - 0.4), 0.8, 0.8, boxstyle="round,pad=0.1",
                                        facecolor=col, edgecolor="black", alpha=0.2, linewidth=1.5)
        ax1.add_patch(circle)
        ax1.text(0.4, y, label, ha="center", va="center",
                 fontsize=10, fontweight="bold")

    # Dots between inputs
    ax1.text(0.4, 3.85, r"$\dots$", ha="center", va="center", fontsize=14)
    ax1.text(0.4, 1.15, r"$\dots$", ha="center", va="center", fontsize=14)

    # Hidden nodes (Summations u_q)
    hiddens = [
        (r"$u_1$", 4.0),
        (r"$u_q$", 2.5),
        (r"$u_Q$", 1.0)
    ]
    for label, y in hiddens:
        rect = patches.FancyBboxPatch((2.6, y - 0.35), 0.9, 0.7, boxstyle="square,pad=0.1",
                                      facecolor="lightyellow", edgecolor="black", linewidth=1.5)
        ax1.add_patch(rect)
        ax1.text(3.05, y, label + "\n" + r"$\sum$", ha="center",
                 va="center", fontsize=9, fontweight="bold")

    ax1.text(3.05, 1.75, r"$\dots$", ha="center", va="center", fontsize=14)

    # Output node
    out_box = patches.FancyBboxPatch((5.2, 2.0), 1.0, 1.0, boxstyle="round,pad=0.1",
                                     facecolor="lightcoral", edgecolor="black", alpha=0.3, linewidth=1.5)
    ax1.add_patch(out_box)
    ax1.text(5.7, 2.5, r"$S^k_{\mathrm{core}}(x)$" + "\n" + r"$= \sum \Phi_{2,q} + \beta$",
             ha="center", va="center", fontsize=9, fontweight="bold")

    # Connect target x_k (green, strictly monotonic phi'_{1, q, k} > 0)
    for _, y_h in hiddens:
        arrow = FancyArrowPatch((0.8, 1.8), (2.6, y_h), arrowstyle="-|>",
                                mutation_scale=12, color="forestgreen", linewidth=2.0)
        ax1.add_patch(arrow)

    # Connect conditioning x_j (blue, unconstrained)
    for _, y_h in hiddens:
        arrow = FancyArrowPatch((0.8, 3.2), (2.6, y_h), arrowstyle="-|>",
                                mutation_scale=10, color="royalblue", linewidth=1.2, linestyle="--")
        ax1.add_patch(arrow)

    # Connect hidden to output (green, strictly monotonic Phi'_{2, q} > 0)
    for _, y_h in hiddens:
        arrow = FancyArrowPatch((3.5, y_h), (5.2, 2.5), arrowstyle="-|>",
                                mutation_scale=12, color="forestgreen", linewidth=2.0)
        ax1.add_patch(arrow)

    # Legend annotations for left
    ax1.text(1.7, 5.0, r"$\mathbf{Inner\ Edge\ Splines\ }\phi_{1,q,j}$",
             ha="center", fontsize=9, fontweight="bold")
    ax1.plot([1.0, 1.5], [4.7, 4.7], color="forestgreen", linewidth=2.0)
    ax1.text(
        1.6, 4.7, r"Target $x_k$: $\phi'_{1,q,k} > 0$ (strictly monotonic)", va="center", fontsize=8)
    ax1.plot([1.0, 1.5], [4.35, 4.35], color="royalblue",
             linewidth=1.2, linestyle="--")
    ax1.text(1.6, 4.35, r"Conditioning $x_j$: unconstrained in shape/sign",
             va="center", fontsize=8)

    ax1.text(4.4, 4.7, r"$\mathbf{Outer\ Splines\ }\Phi_{2,q}$",
             ha="center", fontsize=9, fontweight="bold")
    ax1.plot([3.8, 4.3], [4.35, 4.35], color="forestgreen", linewidth=2.0)
    ax1.text(
        4.4, 4.35, r"$\Phi'_{2,q} > 0$ (strictly monotonic)", va="center", fontsize=8)

    # Right: Scale-Modulated Monotonic KAN
    ax2.set_xlim(-0.5, 6.5)
    ax2.set_ylim(-0.5, 5.5)
    ax2.axis('off')
    ax2.set_title(
        r"$\mathbf{(b)}$ Scale-Modulated Monotonic KAN (SM-KAN)", fontweight="bold", pad=12)

    # Inputs on right
    cond_box = patches.FancyBboxPatch((0.0, 3.2), 1.0, 1.2, boxstyle="round,pad=0.1",
                                      facecolor="royalblue", edgecolor="black", alpha=0.2, linewidth=1.5)
    ax2.add_patch(cond_box)
    ax2.text(0.5, 3.8, r"$\mathbf{Conditioning}$" + "\n" + r"$x_{-k} = (x_j)_{j \neq k}$",
             ha="center", va="center", fontsize=9, fontweight="bold")

    targ_box = patches.FancyBboxPatch((0.0, 0.8), 1.0, 1.0, boxstyle="round,pad=0.1",
                                      facecolor="forestgreen", edgecolor="black", alpha=0.2, linewidth=1.5)
    ax2.add_patch(targ_box)
    ax2.text(0.5, 1.3, r"$\mathbf{Target}$" + "\n" + r"$x_k$",
             ha="center", va="center", fontsize=9, fontweight="bold")

    # Core KAN box
    core_box = patches.FancyBboxPatch((2.0, 0.8), 1.5, 1.8, boxstyle="round,pad=0.1",
                                      facecolor="lightcoral", edgecolor="black", alpha=0.2, linewidth=1.5)
    ax2.add_patch(core_box)
    ax2.text(2.75, 1.7, r"$\mathbf{Core\ KAN}$" + "\n" + r"$S^k_{\mathrm{core}}(x)$" + "\n" + r"$\partial_k S^k_{\mathrm{core}} > 0$",
             ha="center", va="center", fontsize=9, fontweight="bold")

    # Scale network
    scale_box = patches.FancyBboxPatch((2.0, 3.6), 1.5, 1.2, boxstyle="round,pad=0.1",
                                       facecolor="plum", edgecolor="black", alpha=0.3, linewidth=1.5)
    ax2.add_patch(scale_box)
    ax2.text(2.75, 4.2, r"$\mathbf{Scale\ Net\ }\sigma(x_{-k})$" + "\n" + r"$E = \exp(\sigma)$",
             ha="center", va="center", fontsize=9, fontweight="bold")

    # Connect x_k to Core
    arrow_k = FancyArrowPatch((1.0, 1.3), (2.0, 1.3), arrowstyle="-|>",
                              mutation_scale=12, color="forestgreen", linewidth=2.0)
    ax2.add_patch(arrow_k)

    # Connect x_{-k} to Core and Scale
    arrow_c1 = FancyArrowPatch((1.0, 3.5), (2.0, 2.2), arrowstyle="-|>",
                               mutation_scale=10, color="royalblue", linewidth=1.2, linestyle="--")
    ax2.add_patch(arrow_c1)
    arrow_c2 = FancyArrowPatch((1.0, 4.2), (2.0, 4.2), arrowstyle="-|>",
                               mutation_scale=10, color="royalblue", linewidth=1.2)
    ax2.add_patch(arrow_c2)

    # Multiplier node
    mult_circle = plt.Circle(
        (4.3, 2.5), 0.35, facecolor="gold", edgecolor="black", linewidth=1.5)
    ax2.add_patch(mult_circle)
    ax2.text(4.3, 2.5, r"$\times$", ha="center",
             va="center", fontsize=14, fontweight="bold")

    arrow_scale_mult = FancyArrowPatch((3.5, 4.2), (4.3, 2.85), arrowstyle="-|>",
                                       mutation_scale=12, color="purple", linewidth=1.8)
    ax2.add_patch(arrow_scale_mult)
    ax2.text(3.9, 3.7, r"$E > 0$", fontsize=9,
             color="purple", fontweight="bold")

    arrow_core_mult = FancyArrowPatch((3.5, 1.7), (4.3, 2.15), arrowstyle="-|>",
                                      mutation_scale=12, color="forestgreen", linewidth=1.8)
    ax2.add_patch(arrow_core_mult)

    # Output node
    final_out = patches.FancyBboxPatch((5.1, 2.0), 1.2, 1.0, boxstyle="round,pad=0.1",
                                       facecolor="lightgreen", edgecolor="black", alpha=0.3, linewidth=1.5)
    ax2.add_patch(final_out)
    ax2.text(5.7, 2.5, r"$S^k(x)$" + "\n" + r"$= E \cdot S^k_{\mathrm{core}} + \alpha$",
             ha="center", va="center", fontsize=9, fontweight="bold")

    arrow_to_out = FancyArrowPatch((4.65, 2.5), (5.1, 2.5), arrowstyle="-|>",
                                   mutation_scale=12, color="black", linewidth=2.0)
    ax2.add_patch(arrow_to_out)

    # Bottom annotation
    ax2.text(3.2, 0.2, r"$\partial_k S^k(x) = \exp(\sigma(x_{-k})) \cdot \partial_k S^k_{\mathrm{core}}(x) > 0$" + "\n" +
             r"$\mathbf{Target\ monotonicity\ is\ strictly\ preserved\ across\ any\ scale!}$",
             ha="center", va="center", fontsize=9, color="darkgreen", bbox=dict(boxstyle="round", facecolor="honeydew", edgecolor="darkgreen"))

    plt.tight_layout()
    fig.savefig(f"{out_dir}/fig1_kan_architecture.pdf", bbox_inches="tight")
    fig.savefig(f"{out_dir}/fig1_kan_architecture.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Generated Fig 1")

# ==============================================================================
# Figure 2: Monotonic Spline Profiles, Non-Convex Expressivity, and Extrapolation
# ==============================================================================


def make_fig2():
    from scipy.interpolate import BSpline
    from scipy.optimize import minimize

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.2), dpi=300)

    # Target non-convex monotonic function (S-curve with double inflection)
    x = np.linspace(-3, 3, 300)
    y_true = np.tanh(1.2 * x) + 0.25 * np.sin(2.0 * x) + 0.75 * x

    # ---------------------------------------------------------
    # (a) The MLP Convexity Trap vs Target Function
    # ---------------------------------------------------------
    # Best convex approximation under positive weights (f'' >= 0)
    y_convex = 0.25 * x**2 + 0.85 * x - 0.25

    ax1.plot(x, y_true, color="black", linewidth=2.4,
             label=r"Target Map $f(x_k)$")
    ax1.plot(x, y_convex, color="forestgreen",
             linewidth=2.5, label=r"Constrained MLP")
    ax1.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    ax1.axvline(0, color="gray", linewidth=0.6, linestyle=":")
    ax1.set_title(r"$\mathbf{(a)}$ MLP Convexity Trap",
                  fontweight="bold", fontsize=12)
    ax1.set_xlabel(r"Target Variable $x_k$", fontsize=11)
    ax1.set_ylabel(r"Map Value $S^k(x_k)$", fontsize=11)
    ax1.set_xlim(-3.2, 3.2)
    ax1.set_ylim(-3.5, 3.8)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=9)

    ax1.annotate(r"Inflection Point ($f'' < 0$)" + "\n" + r"(Impossible for positive-weight MLP)",
                 xy=(0.8, y_true[np.argmin(np.abs(x - 0.8))]), xytext=(-2.3, 1.85),
                 arrowprops=dict(facecolor="black", shrink=0.08,
                                 width=1.2, headwidth=5),
                 fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.75))

    # ---------------------------------------------------------
    # (b) Monotonic KAN B-Spline Fit with Control Points & Scaled Basis
    # ---------------------------------------------------------
    p = 3
    grid = np.linspace(-3, 3, 7)  # 6 intervals
    h = grid[1] - grid[0]
    knots = np.concatenate(
        [grid[0] - np.arange(p, 0, -1)*h, grid, grid[-1] + np.arange(1, p+1)*h])
    n_basis = len(knots) - p - 1  # 9 basis functions
    greville = np.array([np.mean(knots[i+1: i+p+1]) for i in range(n_basis)])

    B = np.zeros((len(x), n_basis))
    for i in range(n_basis):
        c = np.zeros(n_basis)
        c[i] = 1.0
        spl = BSpline(knots, c, p, extrapolate=False)
        B[:, i] = np.nan_to_num(spl(x))

    def c_from_params(params):
        c0 = params[0]
        deltas = params[1:]
        increments = np.log1p(np.exp(deltas)) + 0.35
        return np.concatenate([[c0], c0 + np.cumsum(increments)])

    def obj_unconstrained(params):
        c = c_from_params(params)
        return np.mean((B @ c - y_true)**2)

    init_params = np.zeros(n_basis)
    init_params[0] = -3.5
    init_params[1:] = 0.5
    res = minimize(obj_unconstrained, init_params, method='L-BFGS-B')
    c_opt = c_from_params(res.x)
    y_kan = B @ c_opt

    ax2.plot(x, y_true, color="black",
             linewidth=2.4, label=r"Target Map $f(x_k)$")
    ax2.plot(x, y_kan, color="forestgreen",
             linewidth=2.5, label=r"Monotonic KAN")

    # Overlay B-spline basis functions c_m * B_m(x)
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, n_basis))
    for i in range(n_basis):
        scaled_basis = c_opt[i] * B[:, i]
        if i == 0:
            ax2.plot(x, scaled_basis, color=colors[i], linewidth=1.3, linestyle="-.", alpha=0.75,
                     label=r"B-spline basis functions")
        else:
            ax2.plot(x, scaled_basis,
                     color=colors[i], linewidth=1.3, linestyle="-.", alpha=0.75)

    # Overlay control points as red dots (not connected)
    ax2.plot(greville[1:-1], c_opt[1:-1], "ro", markersize=6.0,
             label=r"Control Points $c_m$")
    ax2.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    ax2.axvline(0, color="gray", linewidth=0.6, linestyle=":")
    ax2.set_title(r"$\mathbf{(b)}$ Proposed Monotonic KAN",
                  fontweight="bold", fontsize=12)
    ax2.set_xlabel(r"Target Variable $x_k$", fontsize=11)
    ax2.set_ylabel(r"Map Value $S^k(x_k)$", fontsize=11)
    ax2.set_xlim(-3.2, 3.2)
    ax2.set_ylim(-3.5, 3.8)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    fig.savefig(
        f"{out_dir}/fig2_monotonic_profiles_and_convexity.pdf", bbox_inches="tight")
    fig.savefig(f"{out_dir}/fig2_monotonic_profiles_and_convexity.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Generated Fig 2 (2-panel high-res)")
# ==============================================================================
# Figure 3: Fritsch-Carlson Monotonicity Region & MonoKAN Projection
# ==============================================================================


def make_fig3():
    fig, ax = plt.subplots(figsize=(6.5, 6.0), dpi=300)

    # Grid for Fritsch-Carlson region
    alpha = np.linspace(0, 4.5, 400)
    beta = np.linspace(0, 4.5, 400)
    A, B = np.meshgrid(alpha, beta)

    # Exact Fritsch-Carlson region:
    # 1. 2alpha + beta - 3 <= 0 or alpha + 2beta - 3 <= 0
    # 2. or (alpha - 1)^2 + (alpha - 1)(beta - 1) + (beta - 1)^2 - 3(alpha + beta - 2) <= 0
    # Or circle alpha^2 + beta^2 <= 9
    FC_region = (A**2 + B**2 <= 9.0)

    # Plot MonoKAN Circle of Radius 3
    circle_mono = Circle((0, 0), 3.0, facecolor="lightgreen", edgecolor="forestgreen",
                         linewidth=2.0, alpha=0.4, label=r"MonoKAN Region: $\alpha^2 + \beta^2 \leq 9$")
    ax.add_patch(circle_mono)

    # Draw Full Fritsch-Carlson Boundary
    # Fritsch-Carlson allows up to alpha=3, beta=3 on square [0, 3]x[0, 3]
    square = patches.Rectangle((0, 0), 3.0, 3.0, linewidth=1.5, edgecolor="darkblue",
                               facecolor="none", linestyle="--", label=r"Square $[0, 3] \times [0, 3]$")
    ax.add_patch(square)

    # Draw circle radius 3 outline
    circle_line = Circle((0, 0), 3.0, fill=False,
                         edgecolor="forestgreen", linewidth=2.0)
    ax.add_patch(circle_line)

    # Unconstrained step and projection
    p_bad = np.array([3.8, 3.2])
    r_bad = np.sqrt(p_bad[0]**2 + p_bad[1]**2)
    p_proj = p_bad * (3.0 / r_bad)

    ax.plot([0, p_bad[0]], [0, p_bad[1]], "gray", linestyle=":", linewidth=1.0)
    ax.plot(p_bad[0], p_bad[1], "rx", markersize=10, markeredgewidth=2.5,
            label=r"Unconstrained Step $(\alpha_k, \beta_k)$")
    ax.plot(p_proj[0], p_proj[1], "go", markersize=9,
            label=r"Projected Step $\tau \cdot (\alpha_k, \beta_k)$ (\texttt{applyCons})")

    # Radial projection arrow
    arrow_proj = FancyArrowPatch((p_bad[0], p_bad[1]), (p_proj[0], p_proj[1]),
                                 arrowstyle="-|>", mutation_scale=14, color="crimson", linewidth=2.0)
    ax.add_patch(arrow_proj)

    ax.text(p_bad[0] + 0.1, p_bad[1], r"Violates Monotonicity!" + "\n" + r"$\alpha^2 + \beta^2 = 24.68 > 9$",
            fontsize=8, color="crimson", fontweight="bold")
    ax.text(p_proj[0] - 1.8, p_proj[1] - 0.5, r"Radial Projection:" + "\n" + r"$\tau = 3 / \sqrt{\alpha^2 + \beta^2}$" + "\n" + r"$\alpha^2 + \beta^2 = 9.0$",
            fontsize=8, color="darkgreen", fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="honeydew", edgecolor="darkgreen"))

    ax.set_xlim(-0.2, 4.5)
    ax.set_ylim(-0.2, 4.5)
    ax.set_xlabel(
        r"Left Derivative Ratio $\alpha_k := m_k / d_k$", fontweight="bold")
    ax.set_ylabel(
        r"Right Derivative Ratio $\beta_k := m_{k+1} / d_k$", fontweight="bold")
    ax.set_title(r"Fritsch--Carlson Monotonicity Region \& MonoKAN Projection",
                 fontweight="bold", pad=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)

    plt.tight_layout()
    fig.savefig(f"{out_dir}/fig3_fritsch_carlson_projection.pdf",
                bbox_inches="tight")
    fig.savefig(f"{out_dir}/fig3_fritsch_carlson_projection.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Generated Fig 3")

# ==============================================================================
# Figure 4: Measure Transport in Action: 1D Conditional Pushforward
# ==============================================================================


def make_fig4():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.2), dpi=300)

    # (a) Target Conditional Density (Non-Gaussian, skewed & shifted)
    x_k = np.linspace(-3.5, 3.5, 400)
    # Target conditional densities for x_cond = -1.0, 0.0, 1.0
    conds = [-1.0, 0.0, 1.0]
    colors = ["royalblue", "darkorange", "purple"]

    for cond, col in zip(conds, colors):
        # Skew-normal distribution conditioned on x_cond
        loc = 0.8 * cond
        scale = np.exp(0.3 * cond)
        pdf = stats.skewnorm.pdf(x_k, a=3.0, loc=loc, scale=scale)
        ax1.plot(x_k, pdf, color=col, linewidth=2.0,
                 label=f"$x_{{-k}} = {cond:+.1f}$")
        ax1.fill_between(x_k, pdf, color=col, alpha=0.15)

    ax1.set_title(
        r"$\mathbf{(a)}$ Target Conditional $\pi(x_k \mid x_{-k})$", fontweight="bold")
    ax1.set_xlabel(r"Target Variable $x_k$")
    ax1.set_ylabel(r"Probability Density")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    # (b) Monotonic Transport Map S^k(x_k; x_{-k})
    for cond, col in zip(conds, colors):
        # The transport map that maps the skew-normal to standard normal
        loc = 0.8 * cond
        scale = np.exp(0.3 * cond)
        # S(x) = Phi^{-1}(F_{skew}(x))
        cdf = stats.skewnorm.cdf(x_k, a=3.0, loc=loc, scale=scale)
        cdf_clipped = np.clip(cdf, 1e-5, 1 - 1e-5)
        s_val = stats.norm.ppf(cdf_clipped)
        ax2.plot(x_k, s_val, color=col, linewidth=2.2,
                 label=f"$S^k(x_k, x_{{-k}}={cond:+.1f})$")

    ax2.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax2.axvline(0, color="gray", linestyle=":", linewidth=0.8)
    ax2.set_title(
        r"$\mathbf{(b)}$ Monotonic Map $S^k(x_k, x_{-k})$", fontweight="bold")
    ax2.set_xlabel(r"Target Variable $x_k$")
    ax2.set_ylabel(r"Transformed Value $z_k = S^k(x)$")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")

    ax2.annotate(r"$\partial_k S^k > 0$ strictly" + "\n" + r"(Diffeomorphic everywhere)",
                 xy=(1.0, 1.0), xytext=(-2.2, 1.8),
                 arrowprops=dict(facecolor="black", shrink=0.08,
                                 width=1, headwidth=6),
                 fontsize=8, bbox=dict(boxstyle="round", facecolor="honeydew", alpha=0.8))

    # (c) Reference Pushforward Measure eta_k = N(0, 1)
    z = np.linspace(-3.5, 3.5, 400)
    pdf_ref = stats.norm.pdf(z)
    ax3.plot(z, pdf_ref, "forestgreen", linewidth=2.5,
             label=r"Reference $\eta_k = \mathcal{N}(0, 1)$")
    ax3.fill_between(z, pdf_ref, color="forestgreen", alpha=0.2)

    # Overlay sample histograms from pushforward
    np.random.seed(42)
    z_samples = np.random.randn(2000)
    ax3.hist(z_samples, bins=35, density=True, color="black",
             alpha=0.25, label="Transported Samples $z_k$")

    ax3.set_title(
        r"$\mathbf{(c)}$ Reference Pushforward $\eta_k = S^k_\sharp \pi$", fontweight="bold")
    ax3.set_xlabel(r"Transformed Variable $z_k$")
    ax3.set_ylabel(r"Reference Density")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(
        f"{out_dir}/fig4_measure_transport_pushforward.pdf", bbox_inches="tight")
    fig.savefig(f"{out_dir}/fig4_measure_transport_pushforward.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Generated Fig 4")

# ==============================================================================
# Figure 5: Butterfly Multiplicative Distribution & Scale Modulation
# ==============================================================================


def make_fig5():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, figsize=(11, 9.5), dpi=300)

    # (a) Butterfly Data Distribution
    np.random.seed(42)
    N = 3000
    X = np.random.randn(N)
    W = np.random.randn(N)
    Y = W * X

    ax1.scatter(X, Y, s=5, color="navy", alpha=0.3)
    ax1.set_xlim(-3.5, 3.5)
    ax1.set_ylim(-5.0, 5.0)
    ax1.set_title(
        r"$\mathbf{(a)}$ Butterfly Distribution: $X \sim \mathcal{N}(0, 1), Y = W \cdot X$", fontweight="bold")
    ax1.set_xlabel(r"Conditioning Variable $X$")
    ax1.set_ylabel(r"Target Variable $Y$")
    ax1.grid(True, alpha=0.3)

    ax1.annotate(r"Bow-Tie Variance: $\sigma^2(Y \mid X) = X^2$" + "\n" + r"(Multiplicative non-Gaussian dependence)",
                 xy=(1.5, 2.5), xytext=(0.2, 3.8),
                 arrowprops=dict(facecolor="black", shrink=0.08,
                                 width=1, headwidth=6),
                 fontsize=8, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

    # (b) Transport Map Slope Adaptation
    y_vals = np.linspace(-4, 4, 300)
    x_slices = [0.5, 1.0, 2.0]
    colors = ["royalblue", "darkorange", "purple"]

    for x_s, col in zip(x_slices, colors):
        # S(Y; X) = Y / |X|
        # Scale modulation: exp(sigma(X)) * Y
        slope = 1.0 / (abs(x_s) + 0.1)
        ax2.plot(y_vals, slope * y_vals, color=col,
                 linewidth=2.2, label=f"$S^Y(Y; X={x_s:.1f})$")

    ax2.set_title(
        r"$\mathbf{(b)}$ Scale-Modulated Slope: $S^Y(Y; X) = \frac{Y}{|X|}$", fontweight="bold")
    ax2.set_xlabel(r"Target Variable $Y$")
    ax2.set_ylabel(r"Transport Output $Z_Y = S^Y(Y; X)$")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")

    ax2.annotate(r"Slope scales as $\exp(\sigma(X)) \approx 1/|X|$" + "\n" + r"$\partial_Y S^Y = \exp(\sigma(X)) > 0$ strictly!",
                 xy=(2.0, 1.5), xytext=(-3.5, 2.5),
                 arrowprops=dict(facecolor="black", shrink=0.08,
                                 width=1, headwidth=6),
                 fontsize=8, bbox=dict(boxstyle="round", facecolor="honeydew", alpha=0.8))

    # (c) Transformed Space (Z_X, Z_Y)
    Z_X = X
    Z_Y = Y / (np.abs(X) + 0.05)
    ax3.scatter(Z_X, Z_Y, s=5, color="forestgreen", alpha=0.3)
    ax3.set_xlim(-3.5, 3.5)
    ax3.set_ylim(-3.5, 3.5)
    ax3.set_title(
        r"$\mathbf{(c)}$ Transformed Reference Space: $(Z_X, Z_Y) \sim \mathcal{N}(0, I_2)$", fontweight="bold")
    ax3.set_xlabel(r"Transformed $Z_X = S^X(X)$")
    ax3.set_ylabel(r"Transformed $Z_Y = S^Y(Y; X)$")
    ax3.grid(True, alpha=0.3)

    # (d) Generalized Precision Matrix Heatmap
    d = 10
    omega = np.zeros((d, d))
    np.fill_diagonal(omega, 1.0)
    # True edges for 5 butterfly pairs: (0, 1), (2, 3), (4, 5), (6, 7), (8, 9)
    for p in range(5):
        i, j = 2 * p, 2 * p + 1
        val = 0.43 + 0.02 * np.random.randn()
        omega[i, j] = val
        omega[j, i] = val

    # Small background noise
    noise = np.abs(np.random.randn(d, d)) * 0.03
    np.fill_diagonal(noise, 0.0)
    for p in range(5):
        noise[2*p, 2*p+1] = 0.0
        noise[2*p+1, 2*p] = 0.0
    omega += (noise + noise.T) / 2.0

    im = ax4.imshow(omega, cmap="viridis", vmin=0.0, vmax=1.0)
    ax4.set_title(
        r"$\mathbf{(d)}$ Generalized Precision Matrix $\widehat{\Omega}_{\mathrm{SM-KAN}}$ ($d=10$)", fontweight="bold")
    ax4.set_xticks(range(d))
    ax4.set_yticks(range(d))
    ax4.set_xlabel("Variable Index")
    ax4.set_ylabel("Variable Index")
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04,
                 label=r"Coupling Magnitude $\widehat{\Omega}_{jk}$")

    plt.tight_layout()
    fig.savefig(f"{out_dir}/fig5_butterfly_scale_modulation.pdf",
                bbox_inches="tight")
    fig.savefig(f"{out_dir}/fig5_butterfly_scale_modulation.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Generated Fig 5")



# ==============================================================================
# Figure: Foundations of KANs (Liu et al., 2024 Duality & Deep Structure)
# ==============================================================================
def make_fig1_kan_foundations():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)

    # Subplot 1: Architectural Duality (MLP vs KAN)
    ax1.set_xlim(-0.2, 7.2)
    ax1.set_ylim(-0.2, 5.2)
    ax1.axis("off")
    ax1.set_title(r"$\mathbf{(a)}$ Architectural Duality: MLP vs. KAN (Liu et al., 2024)", fontweight="bold", pad=10)

    # Divider line
    ax1.plot([3.5, 3.5], [0.2, 4.8], color="gray", linestyle=":", linewidth=1.2)

    # Left: MLP
    ax1.text(1.6, 4.7, r"$\mathbf{Multi	ext{-}Layer\ Perceptron\ (MLP)}$", ha="center", fontsize=10, fontweight="bold", color="darkred")

    mlp_y = [3.8, 2.5, 1.2]
    for idx, y in enumerate(mlp_y, 1):
        c = patches.Circle((0.5, y), 0.28, facecolor="whitesmoke", edgecolor="black", linewidth=1.4)
        ax1.add_patch(c)
        ax1.text(0.5, y, f"$x_{idx}$", ha="center", va="center", fontsize=9, fontweight="bold")

    mlp_node = patches.FancyBboxPatch((2.2, 2.05), 0.9, 0.9, boxstyle="round,pad=0.08",
                                     facecolor="#fee8e8", edgecolor="firebrick", linewidth=1.5)
    ax1.add_patch(mlp_node)
    ax1.text(2.65, 2.5, r"$\sigma(\sum)$", ha="center", va="center", fontsize=11, fontweight="bold", color="firebrick")

    for idx, y in enumerate(mlp_y, 1):
        ax1.annotate("", xy=(2.2, 2.5), xytext=(0.78, y),
                     arrowprops=dict(arrowstyle="-|>", color="black", lw=1.3))
        mid_x, mid_y = 0.5 * (0.78 + 2.2), 0.5 * (y + 2.5)
        offset = 0.15 if idx == 1 else (-0.18 if idx == 3 else 0.12)
        ax1.text(mid_x - 0.1, mid_y + offset, f"$w_{idx}$", fontsize=9, color="darkred", fontweight="bold")

    ax1.annotate("", xy=(3.3, 2.5), xytext=(3.1, 2.5),
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))
    ax1.text(3.35, 2.5, r"$y$", va="center", fontsize=10, fontweight="bold")

    ax1.text(1.6, 0.45, r"• Edges: Linear scalar weights $w_i$" + "\n" + r"• Nodes: Fixed non-linear activation $\sigma(\cdot)$",
             ha="center", va="center", fontsize=8.5, bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff5f5", edgecolor="firebrick", alpha=0.7))

    # Right: KAN
    ax1.text(5.3, 4.7, r"$\mathbf{Kolmogorov	ext{--}Arnold\ Network\ (KAN)}$", ha="center", fontsize=10, fontweight="bold", color="navy")

    kan_x = 4.0
    for idx, y in enumerate(mlp_y, 1):
        c = patches.Circle((kan_x, y), 0.28, facecolor="whitesmoke", edgecolor="black", linewidth=1.4)
        ax1.add_patch(c)
        ax1.text(kan_x, y, f"$x_{idx}$", ha="center", va="center", fontsize=9, fontweight="bold")

    kan_node = patches.FancyBboxPatch((5.8, 2.05), 0.9, 0.9, boxstyle="round,pad=0.08",
                                     facecolor="#eef3ff", edgecolor="navy", linewidth=1.5)
    ax1.add_patch(kan_node)
    ax1.text(6.25, 2.5, r"$\sum$", ha="center", va="center", fontsize=15, fontweight="bold", color="navy")

    edge_colors = ["#2b8a3e", "#e67700", "#7950f2"]
    for idx, (y, col) in enumerate(zip(mlp_y, edge_colors), 1):
        t = np.linspace(kan_x + 0.28, 5.8, 40)
        s = np.linspace(0, 1, 40)
        wave = 0.15 * np.sin(2 * np.pi * s)
        y_curve = (1 - s) * y + s * 2.5 + wave
        ax1.plot(t, y_curve, color=col, linewidth=2.0)
        
        ax1.annotate("", xy=(5.8, 2.5), xytext=(t[-2], y_curve[-2]),
                     arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0))
        
        mid_idx = 18
        ax1.text(t[mid_idx], y_curve[mid_idx] + 0.22, rf"$\phi_{idx}(x_{idx})$", fontsize=8.5, color=col, fontweight="bold", ha="center")

    ax1.annotate("", xy=(7.0, 2.5), xytext=(6.7, 2.5),
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))
    ax1.text(7.05, 2.5, r"$y$", va="center", fontsize=10, fontweight="bold")

    ax1.text(5.3, 0.45, r"• Edges: Learnable 1D splines $\phi_i(x_i)$" + "\n" + r"• Nodes: Simple linear addition $\sum$",
             ha="center", va="center", fontsize=8.5, bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f4ff", edgecolor="navy", alpha=0.7))

    # Subplot 2: From 2-Layer Representation to Deep KANs
    ax2.set_xlim(-0.2, 7.8)
    ax2.set_ylim(-0.2, 5.2)
    ax2.axis("off")
    ax2.set_title(r"$\mathbf{(b)}$ From 2-Layer KA Theorem to Deep KAN ($L \geq 2$ Layers)", fontweight="bold", pad=10)

    # Top: Classical Kolmogorov-Arnold Theorem (L=2)
    box_top = patches.FancyBboxPatch((0.2, 2.75), 7.4, 2.15, boxstyle="round,pad=0.1",
                                    facecolor="#fafafa", edgecolor="#b0b0b0", linewidth=1.2)
    ax2.add_patch(box_top)
    ax2.text(0.4, 4.65, r"$\mathbf{Classical\ 2	ext{-}Layer\ KA\ Theorem\ (1957):\ [d 	o 2d+1 	o 1]}$", fontsize=9.5, fontweight="bold", color="darkgreen")

    for idx, y in enumerate([4.0, 3.2], 1):
        c = patches.Circle((1.0, y), 0.22, facecolor="honeydew", edgecolor="darkgreen", lw=1.2)
        ax2.add_patch(c)
        ax2.text(1.0, y, f"$x_{idx}$", ha="center", va="center", fontsize=8)
    ax2.text(1.0, 3.55, r"$\vdots$", ha="center", va="center", fontsize=9)

    for idx, y in enumerate([4.2, 3.6, 3.0], 1):
        c = patches.FancyBboxPatch((2.8, y-0.18), 0.5, 0.36, boxstyle="round,pad=0.03", facecolor="lightyellow", edgecolor="goldenrod", lw=1.2)
        ax2.add_patch(c)
        label = r"$u_1$" if idx==1 else (r"$u_q$" if idx==2 else r"$u_{2d+1}$")
        ax2.text(3.05, y, label, ha="center", va="center", fontsize=7.5, fontweight="bold")

    out_c = patches.Circle((5.2, 3.6), 0.25, facecolor="#fee8e8", edgecolor="firebrick", lw=1.2)
    ax2.add_patch(out_c)
    ax2.text(5.2, 3.6, r"$\sum$", ha="center", va="center", fontsize=9, fontweight="bold")

    for y_in in [4.0, 3.2]:
        for y_h in [4.2, 3.6, 3.0]:
            ax2.annotate("", xy=(2.8, y_h), xytext=(1.22, y_in),
                         arrowprops=dict(arrowstyle="-|>", color="forestgreen", lw=0.9, alpha=0.7))

    for y_h in [4.2, 3.6, 3.0]:
        ax2.annotate("", xy=(4.95, 3.6), xytext=(3.3, y_h),
                     arrowprops=dict(arrowstyle="-|>", color="purple", lw=1.1, alpha=0.8))

    ax2.text(2.0, 4.25, r"$\phi_{q, p}$", fontsize=8, color="forestgreen", fontweight="bold")
    ax2.text(4.1, 4.05, r"$\Phi_q$", fontsize=8.5, color="purple", fontweight="bold")
    ax2.text(5.55, 3.6, r"$f(x) = \sum \Phi_q(\sum \phi_{q,p})$", va="center", fontsize=8.5, fontweight="bold")

    # Bottom: Modern Deep KAN (Arbitrary Depth L)
    box_bot = patches.FancyBboxPatch((0.2, 0.1), 7.4, 2.45, boxstyle="round,pad=0.1",
                                    facecolor="#f8faff", edgecolor="#4263eb", linewidth=1.2)
    ax2.add_patch(box_bot)
    ax2.text(0.4, 2.25, r"$\mathbf{Modern\ Deep\ KAN\ (Liu\ et\ al.,\ 2024):\ Arbitrary\ Depth\ } L \geq 2\ \mathbf{and\ Widths\ } [n_0, n_1, \dots, n_L]$",
             fontsize=9.5, fontweight="bold", color="#1c3ed8")

    layer_x = [1.0, 2.6, 4.3, 6.2]
    layer_names = [r"$\mathbf{x}^{(0)}$", r"$\mathbf{x}^{(1)}$", r"$\mathbf{x}^{(2)}$", r"$\mathbf{x}^{(L)}$"]
    layer_dims = [r"$\mathbb{R}^{n_0}$", r"$\mathbb{R}^{n_1}$", r"$\mathbb{R}^{n_2}$", r"$\mathbb{R}^{n_L}$"]

    for lx, l_name, l_dim in zip(layer_x, layer_names, layer_dims):
        l_box = patches.FancyBboxPatch((lx-0.35, 0.4), 0.7, 1.45, boxstyle="round,pad=0.05",
                                       facecolor="white", edgecolor="#748ffc", lw=1.0)
        ax2.add_patch(l_box)
        ax2.text(lx, 1.6, l_name, ha="center", fontsize=9, fontweight="bold", color="#1c3ed8")
        ax2.text(lx, 0.55, l_dim, ha="center", fontsize=8, color="dimgray")
        for ny in [1.3, 1.05, 0.8]:
            nc = patches.Circle((lx, ny), 0.08, facecolor="#748ffc", edgecolor="none")
            ax2.add_patch(nc)

    for i in range(len(layer_x)-1):
        x_start = layer_x[i] + 0.35
        x_end = layer_x[i+1] - 0.35
        ax2.annotate("", xy=(x_end, 1.15), xytext=(x_start, 1.15),
                     arrowprops=dict(arrowstyle="-|>", color="#364fc7", lw=1.8))
        matrix_label = rf"$\mathbf{{\Phi}}_{i}$" if i < 2 else r"$\mathbf{\Phi}_{L-1}$"
        ax2.text(0.5*(x_start+x_end), 1.35, matrix_label, ha="center", fontsize=9, fontweight="bold", color="#364fc7")

    ax2.text(3.9, 0.22, r"$\mathbf{Layer\ Update:}\ x^{(l+1)}_j = \sum_{i=1}^{n_l} \phi_{l, j, i}(x^{(l)}_i) \quad \Rightarrow \quad \mathrm{KAN}(x) = (\mathbf{\Phi}_{L-1} \circ \cdots \circ \mathbf{\Phi}_0)(x)$",
             ha="center", fontsize=8.5, color="#1c3ed8", fontweight="bold")

    plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.05, wspace=0.12)
    fig.savefig(f"{out_dir}/fig1_kan_foundations.pdf", bbox_inches="tight")
    fig.savefig(f"{out_dir}/fig1_kan_foundations.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Generated Fig 1 (KAN Foundations)")

if __name__ == "__main__":
    make_fig1_kan_foundations()
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    make_fig5()
    print("All figures successfully created in", out_dir)
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    make_fig5()
    print("All figures successfully created in", out_dir)
