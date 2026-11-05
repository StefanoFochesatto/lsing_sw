import os
import numpy as np
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, FancyBboxPatch
from scipy.interpolate import BSpline

# Use real pdflatex via PGF backend for authentic Computer Modern typography
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
    "pgf.preamble": r"\usepackage{amsmath,amssymb,mathtools,bm}",
})

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(out_dir, exist_ok=True)


def eval_bspline_curve(ctrl_pts, n_pts=150):
    """Evaluate a cubic (degree 3) clamped B-spline on [-1, 1] from control points."""
    ctrl = np.asarray(ctrl_pts, dtype=float)
    m = len(ctrl)
    p = 3
    internal_knots = np.linspace(-1.0, 1.0, m - p + 1)
    knots = np.concatenate([
        np.repeat(-1.0, p),
        internal_knots,
        np.repeat(1.0, p),
    ])
    spl = BSpline(knots, ctrl, p)
    t = np.linspace(-1.0, 1.0, n_pts)
    y = spl(t)
    y_mid = 0.5 * (np.max(y) + np.min(y))
    y_span = 0.5 * (np.max(y) - np.min(y))
    if y_span > 1e-8:
        y = 0.82 * (y - y_mid) / y_span
    return t, y


def draw_spline_box(
    ax,
    cx,
    cy,
    size,
    ctrl_pts,
    edge_color,
    fill_color,
    curve_color,
    lw_box=1.5,
    lw_curve=2.2,
    is_monotonic=True,
    highlight_active=False,
):
    """Draw a Liu et al. (2024) style 1D B-spline activation square inset box on an edge."""
    half = 0.5 * size
    x0, y0 = cx - half, cy - half

    shadow = patches.Rectangle(
        (x0 + 0.025, y0 - 0.025),
        size,
        size,
        facecolor="#14532d22" if highlight_active else "#0f172a10",
        edgecolor="none",
        zorder=4,
    )
    ax.add_patch(shadow)

    rect = patches.Rectangle(
        (x0, y0),
        size,
        size,
        facecolor=fill_color,
        edgecolor=edge_color,
        linewidth=lw_box,
        joinstyle="round",
        zorder=5,
    )
    ax.add_patch(rect)

    pad = 0.07 * size
    for frac in [-0.5, 0.0, 0.5]:
        gx = cx + frac * (size - 2 * pad)
        ax.plot(
            [gx, gx],
            [y0 + pad, y0 + size - pad],
            color="#e2e8f0" if frac != 0.0 else "#cbd5e1",
            linewidth=0.55 if frac != 0.0 else 0.70,
            linestyle="-",
            zorder=5.1,
        )
    ax.plot(
        [x0 + pad, x0 + size - pad],
        [cy, cy],
        color="#cbd5e1",
        linewidth=0.70,
        linestyle="-",
        zorder=5.1,
    )

    t, y = eval_bspline_curve(ctrl_pts)
    margin = 0.84 * half
    px = cx + t * margin
    py = cy + y * margin

    if is_monotonic:
        ax.fill_between(
            px,
            y0 + pad,
            py,
            color=curve_color,
            alpha=0.13 if highlight_active else 0.07,
            zorder=5.3,
        )
    else:
        ax.fill_between(
            px,
            cy,
            py,
            color=curve_color,
            alpha=0.075,
            zorder=5.3,
        )

    ax.plot(
        px,
        py,
        color=curve_color,
        linewidth=lw_curve,
        solid_capstyle="round",
        zorder=6,
    )

    ax.scatter(
        [px[0], px[-1]],
        [py[0], py[-1]],
        s=11 if highlight_active else 8,
        color=curve_color,
        zorder=6.2,
    )


def generate_sm_kan_liu_style():
    fig, ax = plt.subplots(figsize=(14.2, 8.2), dpi=300)
    ax.set_aspect("equal")
    ax.set_xlim(-0.55, 14.45)
    ax.set_ylim(-0.75, 10.65)
    ax.axis("off")

    # Color Palette (identical to fig_2layer_kan_liu.pdf & fig_deep_kan_liu.pdf)
    ACTIVE_DARK = "#0f5132"
    ACTIVE_MAIN = "#15803d"
    ACTIVE_FILL = "#dcfce7"
    ACTIVE_BOX_FILL = "#f0fdf4"

    MONO_SOFT_DARK = "#2d6a4f"
    MONO_SOFT_MAIN = "#52b788"
    MONO_SOFT_LINE = "#68b092"
    MONO_SOFT_BOX = "#f7fbf9"

    BLUE_DARK = "#1e3a5f"
    BLUE_MAIN = "#2b6cb0"
    BLUE_LINE = "#4a80b8"
    BLUE_FILL = "#ebf3fb"
    BLUE_BOX_FILL = "#f7faff"

    SLATE_DARK = "#1e293b"
    SLATE_MID = "#475569"
    NODE_SUM_FILL = "#f8fafc"
    NODE_SUM_EDGE = "#334155"

    # Key coordinates
    y_in = 0.55
    r_in = 0.36
    r_node = 0.35
    r_out = 0.38
    s_box = 0.74

    # Input coordinates
    x_tk = 1.65
    x_cj = 6.95
    x_cd = 10.95

    # -------------------------------------------------------------------------
    # MODULE 1 (LEFT): Certified Monotonic Core S^k_core(x)
    # -------------------------------------------------------------------------
    core_bg = FancyBboxPatch(
        (-0.05, 1.50),
        5.05,
        5.05,
        boxstyle="round,pad=0.12,rounding_size=0.22",
        facecolor="#f0fdf4",
        alpha=0.45,
        edgecolor=ACTIVE_MAIN,
        linestyle=(0, (4, 3)),
        linewidth=1.35,
        zorder=1.5,
    )
    ax.add_patch(core_bg)

    # Module 1 title in unobstructed top-right corner inside box
    ax.text(
        4.92,
        6.15,
        r"\textbf{1. Monotonic Core}" + "\n" + r"$\partial_k S^k_{\text{core}}(x) \ge \epsilon > 0$",
        ha="right",
        va="center",
        linespacing=1.35,
        fontsize=9.5,
        color=ACTIVE_DARK,
        zorder=7,
    )

    x_hid_core = [1.35, 3.60]
    y_box1_core = 2.45
    y_hid_core = 3.85
    y_box2_core = 4.98
    x_core_out, y_core_out = 2.05, 6.02

    # Inner splines of Core: Target -> hidden 1 (active q^*) and hidden 2 (Q)
    core_t_boxes = [(0.95, y_box1_core), (2.28, y_box1_core)]
    core_t_ctrls = [
        [-2.6, -0.35, -0.08, 0.0, 0.08, 0.35, 2.6],
        [-2.2, -1.8, -1.2, -0.2, 0.9, 1.8, 2.5],
    ]
    for q_idx, ((bx, by), xh) in enumerate(zip(core_t_boxes, x_hid_core)):
        is_act = (q_idx == 0)
        ecol = ACTIVE_MAIN if is_act else MONO_SOFT_LINE
        bcol = ACTIVE_MAIN if is_act else MONO_SOFT_MAIN
        bfill = ACTIVE_BOX_FILL if is_act else MONO_SOFT_BOX
        ccol = ACTIVE_DARK if is_act else MONO_SOFT_DARK
        lw = 2.55 if is_act else 1.75
        ax.plot([x_tk, bx], [y_in + r_in, by - 0.5 * s_box], color=ecol, lw=lw, zorder=4.2 if is_act else 3.2)
        ax.plot([bx, xh], [by + 0.5 * s_box, y_hid_core - r_node], color=ecol, lw=lw, zorder=4.2 if is_act else 3.2)
        draw_spline_box(
            ax, bx, by, s_box, core_t_ctrls[q_idx],
            edge_color=bcol, fill_color=bfill, curve_color=ccol,
            lw_box=1.9 if is_act else 1.4, lw_curve=2.4 if is_act else 1.95,
            is_monotonic=True, highlight_active=is_act,
        )
    ax.text(
        core_t_boxes[0][0] - 0.5 * s_box - 0.10,
        y_box1_core,
        r"$\phi_{1,q^*,k}$",
        ha="right",
        va="center",
        fontsize=9.6,
        color=ACTIVE_DARK,
        zorder=7,
    )

    # Conditioning x_j -> Core hidden nodes (unconstrained inner conditioning splines)
    core_c_boxes = [(3.32, y_box1_core), (4.28, y_box1_core)]
    core_c_ctrls = [
        [1.8, 0.3, -1.4, -1.7, -0.9, 0.6, 1.9],
        [-1.4, 1.6, 0.2, -1.6, -0.3, 1.7, -0.8],
    ]
    for q_idx, ((bx, by), xh) in enumerate(zip(core_c_boxes, x_hid_core)):
        ax.plot([x_cj, bx], [y_in + r_in, by - 0.5 * s_box], color=BLUE_LINE, lw=1.30, alpha=0.65, zorder=2.4)
        ax.plot([bx, xh], [by + 0.5 * s_box, y_hid_core - r_node], color=BLUE_LINE, lw=1.30, alpha=0.65, zorder=2.4)
        draw_spline_box(
            ax, bx, by, s_box, core_c_ctrls[q_idx],
            edge_color=BLUE_MAIN, fill_color=BLUE_BOX_FILL, curve_color=BLUE_DARK,
            lw_box=1.25, lw_curve=1.85, is_monotonic=False, highlight_active=False,
        )

    # Hidden nodes inside Core
    c_h0 = Circle((x_hid_core[0], y_hid_core), r_node, facecolor=ACTIVE_FILL, edgecolor=ACTIVE_MAIN, linewidth=2.2, zorder=6)
    ax.add_patch(c_h0)
    ax.text(x_hid_core[0], y_hid_core, r"$\bm{\Sigma}$", ha="center", va="center", fontsize=13.5, color=ACTIVE_DARK, zorder=7)
    ax.text(x_hid_core[0] - r_node - 0.14, y_hid_core, r"$x^{(1)}_{q^*}$", ha="right", va="center", fontsize=10.5, color=ACTIVE_DARK, zorder=7)

    c_h1 = Circle((x_hid_core[1], y_hid_core), r_node, facecolor=NODE_SUM_FILL, edgecolor=NODE_SUM_EDGE, linewidth=1.8, zorder=6)
    ax.add_patch(c_h1)
    ax.text(x_hid_core[1], y_hid_core, r"$\bm{\Sigma}$", ha="center", va="center", fontsize=13.5, color=SLATE_DARK, zorder=7)
    ax.text(x_hid_core[1] + r_node + 0.14, y_hid_core, r"$x^{(1)}_Q$", ha="left", va="center", fontsize=10.5, color=SLATE_DARK, zorder=7)

    ax.text(2.48, y_hid_core, r"$\cdots$", ha="center", va="center", fontsize=13.5, color=SLATE_MID, zorder=6)

    # Outer splines of Core
    outer_ctrls = [
        [-2.5, -0.9, 0.2, 1.0, 1.55, 1.95, 2.25],
        [-2.3, -2.0, -1.4, -0.1, 1.4, 2.0, 2.3],
    ]
    for q_idx, xh in enumerate(x_hid_core):
        is_act = (q_idx == 0)
        ecol = ACTIVE_MAIN if is_act else MONO_SOFT_LINE
        bcol = ACTIVE_MAIN if is_act else MONO_SOFT_MAIN
        bfill = ACTIVE_BOX_FILL if is_act else MONO_SOFT_BOX
        ccol = ACTIVE_DARK if is_act else MONO_SOFT_DARK
        lw = 2.55 if is_act else 1.75
        ax.plot([xh, xh], [y_hid_core + r_node, y_box2_core - 0.5 * s_box], color=ecol, lw=lw, zorder=4.2 if is_act else 3.2)
        ax.plot([xh, x_core_out], [y_box2_core + 0.5 * s_box, y_core_out - r_node], color=ecol, lw=lw, zorder=4.2 if is_act else 3.2)
        draw_spline_box(
            ax, xh, y_box2_core, s_box, outer_ctrls[q_idx],
            edge_color=bcol, fill_color=bfill, curve_color=ccol,
            lw_box=1.9 if is_act else 1.4, lw_curve=2.4 if is_act else 1.95,
            is_monotonic=True, highlight_active=is_act,
        )

    ax.text(x_hid_core[0] - 0.5 * s_box - 0.10, y_box2_core, r"$\phi_{2,1,q^*}$", ha="right", va="center", fontsize=9.6, color=ACTIVE_DARK, zorder=7)
    ax.text(x_hid_core[1] + 0.5 * s_box + 0.10, y_box2_core, r"$\phi_{2,1,Q}$", ha="left", va="center", fontsize=9.6, color=MONO_SOFT_DARK, zorder=7)
    ax.text(2.48, y_box2_core, r"$\cdots$", ha="center", va="center", fontsize=13.5, color=SLATE_MID, zorder=6)

    # Core Output Node S^k_core(x)
    c_core = Circle((x_core_out, y_core_out), r_node, facecolor=ACTIVE_FILL, edgecolor=ACTIVE_DARK, linewidth=2.3, zorder=6)
    ax.add_patch(c_core)
    ax.text(x_core_out, y_core_out, r"$\bm{\Sigma}$", ha="center", va="center", fontsize=14.0, color=ACTIVE_DARK, zorder=7)
    ax.text(x_core_out - r_node - 0.14, y_core_out + 0.04, r"$S^k_{\text{core}}(x)$",
            ha="right", va="center", fontsize=10.8, color=ACTIVE_DARK, zorder=7)

    # -------------------------------------------------------------------------
    # MODULE 2 (CENTER): Conditioning Scale Pathway \sigma(x_{-k}) -> \exp(\cdot)
    # -------------------------------------------------------------------------
    scale_bg = FancyBboxPatch(
        (5.40, 1.50),
        4.10,
        5.05,
        boxstyle="round,pad=0.12,rounding_size=0.22",
        facecolor="#f8fafc",
        alpha=0.80,
        edgecolor=BLUE_MAIN,
        linestyle=(0, (4, 3)),
        linewidth=1.35,
        zorder=1.5,
    )
    ax.add_patch(scale_bg)
    ax.text(
        9.38,
        6.18,
        r"\textbf{2. Scale Pathway}" + "\n" + r"$E \in (e^{-s_{\max}}, e^{s_{\max}})$",
        ha="right",
        va="center",
        linespacing=1.35,
        fontsize=9.2,
        color=BLUE_DARK,
        zorder=7,
    )

    x_psi_boxes = [6.35, 8.55]
    y_psi_box = 2.50
    x_sigma, y_sigma = 7.45, 4.00
    x_exp, y_exp = 7.45, 5.28

    psi_ctrls = [
        [2.2, 1.1, -0.8, -2.1, -0.8, 1.1, 2.2],     # Smooth -\log|x_1|-like symmetric curve
        [-1.5, 1.4, -0.6, -1.6, 1.3, 0.4, -1.2],    # Unconstrained wave
    ]
    in_cond_xs = [x_cj, x_cd]

    for idx, (xin, bx) in enumerate(zip(in_cond_xs, x_psi_boxes)):
        ax.plot([xin, bx], [y_in + r_in, y_psi_box - 0.5 * s_box], color=BLUE_LINE, lw=1.65, zorder=3.0)
        ax.plot([bx, x_sigma], [y_psi_box + 0.5 * s_box, y_sigma - r_node], color=BLUE_LINE, lw=1.65, zorder=3.0)
        draw_spline_box(
            ax, bx, y_psi_box, s_box, psi_ctrls[idx],
            edge_color=BLUE_MAIN, fill_color=BLUE_BOX_FILL, curve_color=BLUE_DARK,
            lw_box=1.45, lw_curve=2.05, is_monotonic=False, highlight_active=False,
        )

    # Labels beside the \psi boxes (clear of edges)
    ax.text(x_psi_boxes[0] - 0.5 * s_box - 0.10, y_psi_box, r"$\psi_j$", ha="right", va="center", fontsize=10.2, color=BLUE_DARK, zorder=7)
    ax.text(x_psi_boxes[1] + 0.5 * s_box + 0.10, y_psi_box, r"$\psi_d$", ha="left", va="center", fontsize=10.2, color=BLUE_DARK, zorder=7)
    ax.text(7.45, y_psi_box, r"$\cdots$", ha="center", va="center", fontsize=13.5, color=BLUE_MAIN, zorder=6)

    # \sigma(x_{-k}) summation node (label kept cleanly inside Module 2)
    c_sigma = Circle((x_sigma, y_sigma), r_node, facecolor=BLUE_FILL, edgecolor=BLUE_MAIN, linewidth=2.0, zorder=6)
    ax.add_patch(c_sigma)
    ax.text(x_sigma, y_sigma, r"$\bm{\Sigma}$", ha="center", va="center", fontsize=13.5, color=BLUE_DARK, zorder=7)
    ax.text(x_sigma - r_node - 0.12, y_sigma + 0.05, r"$\sigma(x_{-k})$",
            ha="right", va="center", fontsize=10.2, color=BLUE_DARK, zorder=7)
    ax.text(x_sigma + r_node + 0.12, y_sigma + 0.05, r"$\textstyle\sum_{j \neq k} \psi_j(x_j)$",
            ha="left", va="center", fontsize=9.5, color=BLUE_DARK, zorder=7)

    # Arrow from \sigma to \exp(\cdot) box
    ax.annotate(
        "",
        xy=(x_exp, y_exp - 0.35),
        xytext=(x_sigma, y_sigma + r_node),
        arrowprops=dict(arrowstyle="-|>", color=BLUE_MAIN, lw=1.8, mutation_scale=10),
        zorder=5,
    )

    # \exp(\cdot) > 0 block
    # Keep the complete scale definition inside the module with horizontal padding.
    exp_box = FancyBboxPatch(
        (x_exp - 1.85, y_exp - 0.40),
        3.70,
        0.80,
        boxstyle="round,pad=0.06,rounding_size=0.15",
        facecolor=BLUE_BOX_FILL,
        edgecolor=BLUE_DARK,
        linewidth=1.85,
        zorder=6,
    )
    ax.add_patch(exp_box)
    ax.text(
        x_exp,
        y_exp,
        r"$E \coloneqq \exp\left(s_{\max}\tanh\left(\frac{\sigma}{s_{\max}}\right)\right)$",
        ha="center",
        va="center",
        fontsize=9.2,
        color=BLUE_DARK,
        zorder=7,
    )

    # -------------------------------------------------------------------------
    # MODULE 3 (RIGHT): Conditioning Offset Pathway \alpha(x_{-k})
    # -------------------------------------------------------------------------
    offset_bg = FancyBboxPatch(
        (9.85, 1.50),
        4.10,
        5.05,
        boxstyle="round,pad=0.12,rounding_size=0.22",
        facecolor="#f8fafc",
        alpha=0.80,
        edgecolor=BLUE_MAIN,
        linestyle=(0, (4, 3)),
        linewidth=1.35,
        zorder=1.5,
    )
    ax.add_patch(offset_bg)
    ax.text(
        13.82,
        6.18,
        r"\textbf{3. Offset Pathway}" + "\n" + r"$\alpha(x_{-k}) \in \mathbb{R}$",
        ha="right",
        va="center",
        linespacing=1.35,
        fontsize=9.5,
        color=BLUE_DARK,
        zorder=7,
    )

    x_rho_boxes = [10.80, 13.00]
    y_rho_box = 2.55
    x_alpha, y_alpha = 11.90, 4.85

    rho_ctrls = [
        [-1.8, -0.4, 1.3, 1.9, 0.8, -0.9, -1.6],
        [1.7, -1.2, -1.5, 0.5, 1.8, 0.2, -1.4],
    ]

    for idx, (xin, bx) in enumerate(zip(in_cond_xs, x_rho_boxes)):
        ax.plot([xin, bx], [y_in + r_in, y_rho_box - 0.5 * s_box], color=BLUE_LINE, lw=1.65, zorder=3.0)
        ax.plot([bx, x_alpha], [y_rho_box + 0.5 * s_box, y_alpha - r_node], color=BLUE_LINE, lw=1.65, zorder=3.0)
        draw_spline_box(
            ax, bx, y_rho_box, s_box, rho_ctrls[idx],
            edge_color=BLUE_MAIN, fill_color=BLUE_BOX_FILL, curve_color=BLUE_DARK,
            lw_box=1.45, lw_curve=2.05, is_monotonic=False, highlight_active=False,
        )

    ax.text(x_rho_boxes[0] - 0.5 * s_box - 0.10, y_rho_box, r"$\rho_j$", ha="right", va="center", fontsize=10.2, color=BLUE_DARK, zorder=7)
    ax.text(x_rho_boxes[1] + 0.5 * s_box + 0.10, y_rho_box, r"$\rho_d$", ha="left", va="center", fontsize=10.2, color=BLUE_DARK, zorder=7)
    ax.text(11.90, y_rho_box, r"$\cdots$", ha="center", va="center", fontsize=13.5, color=BLUE_MAIN, zorder=6)

    # \alpha(x_{-k}) summation node (label kept cleanly inside Module 3)
    c_alpha = Circle((x_alpha, y_alpha), r_node, facecolor=BLUE_FILL, edgecolor=BLUE_MAIN, linewidth=2.0, zorder=6)
    ax.add_patch(c_alpha)
    ax.text(x_alpha, y_alpha, r"$\bm{\Sigma}$", ha="center", va="center", fontsize=13.5, color=BLUE_DARK, zorder=7)
    ax.text(x_alpha - r_node - 0.12, y_alpha + 0.05, r"$\alpha(x_{-k})$",
            ha="right", va="center", fontsize=10.2, color=BLUE_DARK, zorder=7)
    ax.text(x_alpha + r_node + 0.12, y_alpha + 0.05, r"$\textstyle\sum_{j \neq k} \rho_j(x_j)$",
            ha="left", va="center", fontsize=9.5, color=BLUE_DARK, zorder=7)

    # -------------------------------------------------------------------------
    # 4. MULTIPLICATIVE MODULATION GATE (\times) & FINAL OUTPUT (\Sigma)
    # -------------------------------------------------------------------------
    x_mult, y_mult = 5.00, 7.65
    x_out, y_out = 7.45, 8.85

    # Core S^k_core(x) -> Multiplication gate (\times)
    ax.annotate(
        "",
        xy=(x_mult - r_out * 0.85, y_mult - r_out * 0.50),
        xytext=(x_core_out, y_core_out + r_node),
        arrowprops=dict(arrowstyle="-|>", color=ACTIVE_MAIN, lw=2.4, mutation_scale=12),
        zorder=5,
    )

    # Scale \exp(\sigma(x_{-k})) -> Multiplication gate (\times)
    ax.annotate(
        "",
        xy=(x_mult + r_out * 0.85, y_mult - r_out * 0.50),
        xytext=(x_exp, y_exp + 0.32),
        arrowprops=dict(arrowstyle="-|>", color=BLUE_MAIN, lw=2.2, mutation_scale=12),
        zorder=5,
    )
    ax.text(
        6.15,
        6.96,
        r"$E$",
        ha="center",
        va="center",
        fontsize=11.5,
        color=BLUE_DARK,
        zorder=7,
    )

    # Multiplication Gate Node (\times)
    c_mult = Circle((x_mult, y_mult), r_out, facecolor=ACTIVE_FILL, edgecolor=ACTIVE_DARK, linewidth=2.4, zorder=6)
    ax.add_patch(c_mult)
    ax.text(x_mult, y_mult, r"$\bm{\times}$", ha="center", va="center", fontsize=16.5, color=ACTIVE_DARK, zorder=7)
    ax.text(
        x_mult - r_out - 0.22,
        y_mult + 0.05,
        r"$E \cdot S^k_{\text{core}}(x)$",
        ha="right",
        va="center",
        fontsize=10.6,
        color=ACTIVE_DARK,
        bbox=dict(boxstyle="round,pad=0.28", facecolor=ACTIVE_BOX_FILL, edgecolor=ACTIVE_MAIN, lw=1.1),
        zorder=7,
    )

    # Multiplication gate -> Final Output Summation (\Sigma)
    ax.annotate(
        "",
        xy=(x_out - r_out * 0.88, y_out - r_out * 0.45),
        xytext=(x_mult + r_out * 0.75, y_mult + r_out * 0.65),
        arrowprops=dict(arrowstyle="-|>", color=ACTIVE_DARK, lw=2.5, mutation_scale=12),
        zorder=5,
    )

    # Offset \alpha(x_{-k}) -> Final Output Summation (\Sigma)
    ax.annotate(
        "",
        xy=(x_out + r_out * 0.92, y_out - r_out * 0.32),
        xytext=(x_alpha - 0.20, y_alpha + r_node),
        arrowprops=dict(arrowstyle="-|>", color=BLUE_MAIN, lw=2.1, mutation_scale=12),
        zorder=5,
    )

    # Final Output Summation Node (\Sigma)
    c_out = Circle((x_out, y_out), r_out, facecolor=ACTIVE_FILL, edgecolor=ACTIVE_DARK, linewidth=2.5, zorder=6)
    ax.add_patch(c_out)
    ax.text(x_out, y_out, r"$\bm{\Sigma}$", ha="center", va="center", fontsize=15.5, color=ACTIVE_DARK, zorder=7)

    # Output arrow to top banner
    y_banner = 10.02
    ax.annotate(
        "",
        xy=(x_out, y_banner - 0.42),
        xytext=(x_out, y_out + r_out + 0.04),
        arrowprops=dict(arrowstyle="-|>", color=ACTIVE_DARK, lw=1.9, mutation_scale=11),
        zorder=5,
    )

    ax.text(
        x_out,
        y_banner,
        r"$S^k(x) = E \cdot S^k_{\text{core}}(x) + \alpha(x_{-k}), \qquad \partial_k S^k(x) = E \cdot \partial_k S^k_{\text{core}}(x) \ge e^{-s_{\max}}\epsilon > 0$",
        ha="center",
        va="center",
        fontsize=10.5,
        color=SLATE_DARK,
        bbox=dict(boxstyle="round,pad=0.38", facecolor="#f8fafc", edgecolor=ACTIVE_MAIN, lw=1.4),
        zorder=7,
    )

    # -------------------------------------------------------------------------
    # 5. INPUT NODES & CAPTIONS AT BOTTOM
    # -------------------------------------------------------------------------
    in_specs = [
        (x_tk, r"$x_k$", ACTIVE_FILL, ACTIVE_MAIN, ACTIVE_DARK, 2.4),
        (x_cj, r"$x_j$", BLUE_FILL, BLUE_MAIN, BLUE_DARK, 1.9),
        (x_cd, r"$x_d$", BLUE_FILL, BLUE_MAIN, BLUE_DARK, 1.9),
    ]
    for xc, lbl, fc, ec, tc, lw in in_specs:
        c = Circle((xc, y_in), r_in, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=6)
        ax.add_patch(c)
        ax.text(xc, y_in, lbl, ha="center", va="center", fontsize=13.5, color=tc, zorder=7)

    ax.text(0.5 * (x_cj + x_cd), y_in, r"$\cdots$", ha="center", va="center", fontsize=14.5, color=BLUE_MAIN, zorder=6)

    ax.text(
        x_tk,
        y_in - 0.62,
        r"\textbf{Target} $x_k$ (enters $S^k_{\text{core}}$ only)",
        ha="center",
        va="top",
        fontsize=10.3,
        color=ACTIVE_DARK,
    )

    cond_x0, cond_x1 = x_cj - 0.65, x_cd + 0.65
    yb_cond = y_in - 0.46
    ax.plot([cond_x0, cond_x1], [yb_cond, yb_cond], color=BLUE_MAIN, linewidth=1.25)
    ax.plot([cond_x0, cond_x0], [yb_cond, yb_cond + 0.09], color=BLUE_MAIN, linewidth=1.25)
    ax.plot([cond_x1, cond_x1], [yb_cond, yb_cond + 0.09], color=BLUE_MAIN, linewidth=1.25)
    ax.text(
        0.5 * (cond_x0 + cond_x1),
        y_in - 0.62,
        r"\textbf{Conditioning} $\;x_{-k} = (x_j)_{j \neq k}$ (unconstrained across $S^k_{\text{core}}$, $\sigma$, and $\alpha$)",
        ha="center",
        va="top",
        fontsize=10.3,
        color=BLUE_DARK,
    )

    pdf_path = os.path.join(out_dir, "fig_sm_kan_liu.pdf")
    png_path = os.path.join(out_dir, "fig_sm_kan_liu.png")
    plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0.10)
    plt.savefig(png_path, bbox_inches="tight", pad_inches=0.10)
    plt.close()
    print(f"Successfully generated: {pdf_path} {png_path}")


if __name__ == "__main__":
    generate_sm_kan_liu_style()
