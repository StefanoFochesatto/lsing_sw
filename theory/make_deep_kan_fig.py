import os
import numpy as np
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
from scipy.interpolate import BSpline

# Use real pdflatex via PGF backend for authentic Computer Modern typography
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
    "pgf.preamble": r"\usepackage{amsmath,amssymb,bm}",
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
    """
    Draw a clean Liu et al. (2024) / pykan style 1D spline activation square inset box on an edge.
    """
    half = 0.5 * size
    x0, y0 = cx - half, cy - half

    # Subtle drop shadow (stronger if on active path P^*)
    shadow = patches.Rectangle(
        (x0 + 0.025, y0 - 0.025),
        size,
        size,
        facecolor="#14532d22" if highlight_active else "#0f172a10",
        edgecolor="none",
        zorder=4,
    )
    ax.add_patch(shadow)

    # Main square box
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

    # Subtle internal B-spline knot grid & zero axes
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

    # Evaluate B-spline and map to box coordinates
    t, y = eval_bspline_curve(ctrl_pts)
    margin = 0.84 * half
    px = cx + t * margin
    py = cy + y * margin

    # Subtle shading under/around curve
    if is_monotonic:
        ax.fill_between(
            px,
            y0 + pad,
            py,
            color=curve_color,
            alpha=0.12 if highlight_active else 0.07,
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

    # Subtle spline boundary dots
    ax.scatter(
        [px[0], px[-1]],
        [py[0], py[-1]],
        s=11 if highlight_active else 9,
        color=curve_color,
        zorder=6.2,
    )


def generate_deep_kan_liu_style():
    fig, ax = plt.subplots(figsize=(12.9, 8.6), dpi=300)
    ax.set_aspect("equal")
    ax.set_xlim(-3.85, 14.65)
    ax.set_ylim(-0.65, 12.15)
    ax.axis("off")

    # Color Palette
    # Active path P^* (bold certified strictly increasing path)
    ACTIVE_DARK = "#0f5132"
    ACTIVE_MAIN = "#15803d"
    ACTIVE_FILL = "#dcfce7"
    ACTIVE_BOX_FILL = "#f0fdf4"

    # Secondary downstream non-decreasing paths (\phi' >= 0)
    MONO_SOFT_DARK = "#2d6a4f"
    MONO_SOFT_MAIN = "#52b788"
    MONO_SOFT_LINE = "#68b092"
    MONO_SOFT_BOX = "#f7fcf9"

    # Unconstrained conditioning inputs (layer l = 0, i != k)
    BLUE_DARK = "#1e3a5f"
    BLUE_MAIN = "#2b6cb0"
    BLUE_LINE = "#4a80b8"
    BLUE_FILL = "#ebf3fb"
    BLUE_BOX_FILL = "#f7faff"

    SLATE_DARK = "#1e293b"
    SLATE_MID = "#475569"
    SLATE_LIGHT = "#94a3b8"
    NODE_SUM_FILL = "#f8fafc"
    NODE_SUM_EDGE = "#334155"

    # Vertical levels (bottom-to-top Liu et al. 2024 Fig 0.1d deep KAN layout)
    y_0 = 0.50       # Input layer x^{(0)}
    y_box0 = 2.30    # Layer l=0 splines \Phi_0
    y_1 = 4.25       # Hidden layer 1: x^{(1)}
    y_box1 = 6.00    # Downstream hidden splines \Phi_1 ... \Phi_{L-2}
    y_Lm1 = 7.75     # Penultimate hidden layer: x^{(L-1)}
    y_box2 = 9.25    # Output layer splines \Phi_{L-1}
    y_L = 10.70      # Output node x^{(L)}_1 = S^k(x)

    s0 = 0.76  # Box size in \Phi_0
    s1 = 0.76  # Box size in \Phi_l (middle)
    s2 = 0.80  # Box size in \Phi_{L-1} (top)

    r_in = 0.35
    r_hid = 0.35
    r_out = 0.38

    # Horizontal coordinates
    x_inputs = [1.75, 5.30, 8.85]     # x_k, x_j, x_d
    dx_box0 = 1.00
    x_hid1 = [2.15, 5.30, 8.45]       # x^{(1)}_{i_1^*}, x^{(1)}_q, x^{(1)}_{n_1}
    x_hid2 = [2.15, 5.30, 8.45]       # x^{(L-1)}_{i_{L-1}^*}, x^{(L-1)}_q, x^{(L-1)}_{n_{L-1}}
    x_out = 5.30

    # Horizontal level dashed guide lines (Liu et al. Fig 0.1d style)
    guide_x0, guide_x1 = 0.10, 10.60
    for y_lvl in [y_0, y_1, y_Lm1, y_L]:
        ax.plot(
            [guide_x0, guide_x1],
            [y_lvl, y_lvl],
            color="#cbd5e1",
            linestyle=(0, (3.5, 3.0)),
            linewidth=0.95,
            zorder=1,
        )

    # -------------------------------------------------------------------------
    # 1. LAYER l = 0 (\Phi_0): Target (Strictly Monotonic) & Conditioning (Unconstrained)
    # -------------------------------------------------------------------------
    target_l0_ctrls = [
        [-2.5, -0.35, -0.05, 0.05, 0.35, 0.9, 2.5],   # Active edge (k -> i_1^*): strictly increasing
        [-2.4, -2.1, -1.5, 0.0, 1.5, 2.1, 2.4],       # Target edge (k -> q): strictly increasing
        [-2.1, -1.8, -1.4, -0.8, -0.1, 1.0, 2.5],     # Target edge (k -> n_1): strictly increasing
    ]
    cond_j_l0_ctrls = [
        [1.9, 0.4, -1.3, -1.7, -1.1, 0.5, 2.0],       # Unconstrained U-curve
        [-0.9, 1.7, 0.3, -1.7, -0.2, 1.8, -0.7],      # Unconstrained wave
        [-1.6, -0.5, 1.5, 2.2, 1.3, -0.8, -1.5],      # Unconstrained bell hump
    ]
    cond_d_l0_ctrls = [
        [1.5, -1.6, 0.9, 1.7, -1.5, -0.4, 1.6],       # Unconstrained oscillation
        [2.0, 1.4, 0.3, -0.9, -1.6, -0.5, 0.4],       # Unconstrained decreasing hook
        [-1.5, 1.6, -1.3, -1.5, 1.7, -1.0, 1.2],      # Unconstrained W-curve
    ]
    all_l0_ctrls = [target_l0_ctrls, cond_j_l0_ctrls, cond_d_l0_ctrls]

    for i_idx, x_in_c in enumerate(x_inputs):
        is_target = i_idx == 0
        for q_idx, x_h1_c in enumerate(x_hid1):
            is_active_edge = is_target and (q_idx == 0)
            x_box_c = x_in_c + (q_idx - 1) * dx_box0

            if is_active_edge:
                e_col = ACTIVE_MAIN
                b_col = ACTIVE_MAIN
                b_fill = ACTIVE_BOX_FILL
                c_col = ACTIVE_DARK
                lw_bot, lw_top = 2.6, 2.6
                alpha_e = 1.0
                z_e = 4.2
            elif is_target:
                e_col = MONO_SOFT_MAIN
                b_col = MONO_SOFT_MAIN
                b_fill = MONO_SOFT_BOX
                c_col = MONO_SOFT_DARK
                lw_bot, lw_top = 1.75, 1.75
                alpha_e = 0.88
                z_e = 3.4
            else:
                e_col = BLUE_LINE
                b_col = BLUE_MAIN
                b_fill = BLUE_BOX_FILL
                c_col = BLUE_DARK
                lw_bot, lw_top = 1.35, 1.25
                alpha_e = 0.70
                z_e = 2.4

            # Input node -> bottom of spline box
            ax.plot(
                [x_in_c, x_box_c],
                [y_0 + r_in, y_box0 - 0.5 * s0],
                color=e_col,
                linewidth=lw_bot,
                alpha=alpha_e,
                solid_capstyle="round",
                zorder=z_e,
            )
            # Top of spline box -> bottom of hidden node x^{(1)}
            ax.plot(
                [x_box_c, x_h1_c],
                [y_box0 + 0.5 * s0, y_1 - r_hid],
                color=e_col,
                linewidth=lw_top,
                alpha=alpha_e,
                solid_capstyle="round",
                zorder=z_e,
            )

            draw_spline_box(
                ax,
                x_box_c,
                y_box0,
                s0,
                all_l0_ctrls[i_idx][q_idx],
                edge_color=b_col,
                fill_color=b_fill,
                curve_color=c_col,
                lw_box=1.95 if is_active_edge else (1.45 if is_target else 1.25),
                lw_curve=2.45 if is_active_edge else (2.0 if is_target else 1.85),
                is_monotonic=is_target,
                highlight_active=is_active_edge,
            )

    # -------------------------------------------------------------------------
    # 2. DOWNSTREAM HIDDEN LAYERS (1 <= l <= L-2): 7-Box Row (Liu et al. Fig 0.1d style)
    #    All edges satisfy \phi'_{l,j,i} >= 0; active path P^* satisfies \phi' >= \epsilon_l > 0
    # -------------------------------------------------------------------------
    mid_boxes = [
        (1.55, 0, 0, True,  [-2.4, -1.7, -0.6, 0.1, 0.8, 1.7, 2.4]),     # Active P^*: i_1^* -> i_{L-1}^* (\phi' >= \epsilon_l > 0)
        (2.80, 0, 1, False, [-2.2, -1.2, -0.2, 0.0, 0.2, 1.2, 2.2]),     # i_1^* -> q (non-decreasing plateau)
        (4.05, 1, 0, False, [-2.0, -1.6, -1.2, -0.4, 0.6, 1.5, 2.1]),    # q -> i_{L-1}^* (non-decreasing)
        (5.30, 1, 1, False, [-2.3, -1.9, -1.1, 0.0, 1.1, 1.9, 2.3]),     # q -> q (non-decreasing S-curve)
        (6.55, 1, 2, False, [-2.1, -0.8, 0.0, 0.0, 0.0, 0.8, 2.1]),      # q -> n_{L-1} (non-decreasing flat-center)
        (7.80, 2, 1, False, [-2.2, -1.5, -0.7, 0.1, 0.9, 1.6, 2.2]),     # n_1 -> q (non-decreasing)
        (9.05, 2, 2, False, [-2.0, -1.7, -1.1, -0.2, 0.8, 1.6, 2.2]),    # n_1 -> n_{L-1} (non-decreasing)
    ]

    for bx, s_idx, d_idx, is_act, ctrls in mid_boxes:
        x_s = x_hid1[s_idx]
        x_d = x_hid2[d_idx]
        e_col = ACTIVE_MAIN if is_act else MONO_SOFT_LINE
        b_col = ACTIVE_MAIN if is_act else MONO_SOFT_MAIN
        b_fill = ACTIVE_BOX_FILL if is_act else MONO_SOFT_BOX
        c_col = ACTIVE_DARK if is_act else MONO_SOFT_DARK
        lw_e = 2.6 if is_act else 1.50
        alpha_e = 1.0 if is_act else 0.82
        z_e = 4.2 if is_act else 3.0

        # x^{(1)} node -> bottom of mid spline box
        ax.plot(
            [x_s, bx],
            [y_1 + r_hid, y_box1 - 0.5 * s1],
            color=e_col,
            linewidth=lw_e,
            alpha=alpha_e,
            solid_capstyle="round",
            zorder=z_e,
        )
        # Top of mid spline box -> bottom of x^{(L-1)} node
        ax.plot(
            [bx, x_d],
            [y_box1 + 0.5 * s1, y_Lm1 - r_hid],
            color=e_col,
            linewidth=lw_e,
            alpha=alpha_e,
            solid_capstyle="round",
            zorder=z_e,
        )

        draw_spline_box(
            ax,
            bx,
            y_box1,
            s1,
            ctrls,
            edge_color=b_col,
            fill_color=b_fill,
            curve_color=c_col,
            lw_box=1.95 if is_act else 1.35,
            lw_curve=2.45 if is_act else 1.95,
            is_monotonic=True,
            highlight_active=is_act,
        )

    # -------------------------------------------------------------------------
    # 3. OUTPUT LAYER SPLINES (l = L-1, \Phi_{L-1}): x^{(L-1)} -> x^{(L)}_1 = S^k(x)
    # -------------------------------------------------------------------------
    outer_ctrls = [
        [-2.5, -0.9, 0.2, 1.0, 1.55, 1.95, 2.3],      # Active P^* outer spline (\phi' >= \epsilon_{L-1} > 0)
        [-2.4, -0.3, -0.05, 0.05, 0.3, 0.7, 2.4],     # Secondary outer spline (\phi' >= 0)
        [-2.2, -1.9, -1.3, -0.1, 1.3, 1.9, 2.2],      # Secondary outer spline (\phi' >= 0)
    ]
    outer_labels = [
        r"$\phi_{L,1,i^*_{L-1}}$",
        r"$\phi_{L,1,q}$",
        r"$\phi_{L,1,n_{L-1}}$",
    ]

    for q_idx, x_h2_c in enumerate(x_hid2):
        is_act = q_idx == 0
        e_col = ACTIVE_MAIN if is_act else MONO_SOFT_LINE
        b_col = ACTIVE_MAIN if is_act else MONO_SOFT_MAIN
        b_fill = ACTIVE_BOX_FILL if is_act else MONO_SOFT_BOX
        c_col = ACTIVE_DARK if is_act else MONO_SOFT_DARK
        lw_e = 2.65 if is_act else 1.75

        # Vertical edge: x^{(L-1)} -> bottom of outer spline box
        ax.plot(
            [x_h2_c, x_h2_c],
            [y_Lm1 + r_hid, y_box2 - 0.5 * s2],
            color=e_col,
            linewidth=lw_e,
            solid_capstyle="round",
            zorder=4.2 if is_act else 3.2,
        )

        # Converging edge: top of outer spline box -> output node S^k(x)
        ax.plot(
            [x_h2_c, x_out],
            [y_box2 + 0.5 * s2, y_L - r_out],
            color=e_col,
            linewidth=lw_e,
            solid_capstyle="round",
            zorder=4.2 if is_act else 3.2,
        )

        draw_spline_box(
            ax,
            x_h2_c,
            y_box2,
            s2,
            outer_ctrls[q_idx],
            edge_color=b_col,
            fill_color=b_fill,
            curve_color=c_col,
            lw_box=1.95 if is_act else 1.45,
            lw_curve=2.45 if is_act else 2.05,
            is_monotonic=True,
            highlight_active=is_act,
        )

        # Place active outer spline label to the left of Box 1 so the gap between Box 1 & Box 2 is wide open
        if q_idx in (0, 1):
            ax.text(
                x_h2_c - 0.5 * s2 - 0.20,
                y_box2,
                outer_labels[q_idx],
                ha="right",
                va="center",
                fontsize=10.5,
                color=c_col,
                zorder=7,
            )
        else:
            ax.text(
                x_h2_c + 0.5 * s2 + 0.20,
                y_box2,
                outer_labels[q_idx],
                ha="left",
                va="center",
                fontsize=10.5,
                color=c_col,
                zorder=7,
            )

    # Horizontal ellipsis in Hidden Layer 1, Hidden Layer L-1, and Outer Layer
    for xm_hid, xm_out in [(3.55, 3.25), (6.70, 6.88)]:
        ax.text(xm_hid, y_1, r"$\cdots$", ha="center", va="center", fontsize=14, color=SLATE_MID, zorder=6)
        ax.text(xm_hid, y_Lm1, r"$\cdots$", ha="center", va="center", fontsize=14, color=SLATE_MID, zorder=6)
        ax.text(xm_out, y_box2, r"$\cdots$", ha="center", va="center", fontsize=14, color=SLATE_MID, zorder=6)

    # -------------------------------------------------------------------------
    # 4. NODES: Input Layer, Hidden Layer 1, Hidden Layer L-1, Output Layer
    # -------------------------------------------------------------------------
    # Input Nodes x^{(0)}
    in_specs = [
        (x_inputs[0], r"$x_k$", ACTIVE_FILL, ACTIVE_MAIN, ACTIVE_DARK, 2.4),
        (x_inputs[1], r"$x_j$", BLUE_FILL, BLUE_MAIN, BLUE_DARK, 1.85),
        (x_inputs[2], r"$x_d$", BLUE_FILL, BLUE_MAIN, BLUE_DARK, 1.85),
    ]
    for xc, lbl, fc, ec, tc, lw in in_specs:
        c = Circle((xc, y_0), r_in, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=6)
        ax.add_patch(c)
        ax.text(xc, y_0, lbl, ha="center", va="center", fontsize=13.5, color=tc, zorder=7)

    ax.text(7.08, y_0, r"$\cdots$", ha="center", va="center", fontsize=14.0, color=BLUE_MAIN, zorder=6)

    # Input Captions below nodes
    ax.text(
        x_inputs[0],
        y_0 - 0.65,
        r"\textbf{Target} $x^{(0)}_k = x_k$",
        ha="center",
        va="top",
        fontsize=10.5,
        color=ACTIVE_DARK,
    )

    cond_x0, cond_x1 = 4.65, 9.50
    yb_cond = y_0 - 0.48
    ax.plot([cond_x0, cond_x1], [yb_cond, yb_cond], color=BLUE_MAIN, linewidth=1.2)
    ax.plot([cond_x0, cond_x0], [yb_cond, yb_cond + 0.09], color=BLUE_MAIN, linewidth=1.2)
    ax.plot([cond_x1, cond_x1], [yb_cond, yb_cond + 0.09], color=BLUE_MAIN, linewidth=1.2)
    ax.text(
        0.5 * (cond_x0 + cond_x1),
        y_0 - 0.65,
        r"\textbf{Conditioning} $\;x_{-k} = (x^{(0)}_i)_{i \neq k}$",
        ha="center",
        va="top",
        fontsize=10.5,
        color=BLUE_DARK,
    )

    # Hidden Layer 1 Nodes: x^{(1)}
    hid1_labels = [r"$x^{(1)}_{i^*_1}$", r"$x^{(1)}_q$", r"$x^{(1)}_{n_1}$"]
    for q_idx, (xc, lbl) in enumerate(zip(x_hid1, hid1_labels)):
        is_act = q_idx == 0
        c = Circle(
            (xc, y_1),
            r_hid,
            facecolor=ACTIVE_FILL if is_act else NODE_SUM_FILL,
            edgecolor=ACTIVE_MAIN if is_act else NODE_SUM_EDGE,
            linewidth=2.3 if is_act else 1.85,
            zorder=6,
        )
        ax.add_patch(c)
        ax.text(
            xc,
            y_1,
            r"$\bm{\Sigma}$",
            ha="center",
            va="center",
            fontsize=14.0,
            color=ACTIVE_DARK if is_act else SLATE_DARK,
            zorder=7,
        )
        ax.text(
            xc - r_hid - 0.22,
            y_1 + 0.04,
            lbl,
            ha="right",
            va="center",
            fontsize=11.2,
            color=ACTIVE_DARK if is_act else SLATE_DARK,
            zorder=7,
        )

    # Penultimate Hidden Layer L-1 Nodes: x^{(L-1)}
    hid2_labels = [r"$x^{(L-1)}_{i^*_{L-1}}$", r"$x^{(L-1)}_q$", r"$x^{(L-1)}_{n_{L-1}}$"]
    for q_idx, (xc, lbl) in enumerate(zip(x_hid2, hid2_labels)):
        is_act = q_idx == 0
        c = Circle(
            (xc, y_Lm1),
            r_hid,
            facecolor=ACTIVE_FILL if is_act else NODE_SUM_FILL,
            edgecolor=ACTIVE_MAIN if is_act else NODE_SUM_EDGE,
            linewidth=2.3 if is_act else 1.85,
            zorder=6,
        )
        ax.add_patch(c)
        ax.text(
            xc,
            y_Lm1,
            r"$\bm{\Sigma}$",
            ha="center",
            va="center",
            fontsize=14.0,
            color=ACTIVE_DARK if is_act else SLATE_DARK,
            zorder=7,
        )
        ax.text(
            xc - r_hid - 0.22,
            y_Lm1 + 0.04,
            lbl,
            ha="right",
            va="center",
            fontsize=11.2,
            color=ACTIVE_DARK if is_act else SLATE_DARK,
            zorder=7,
        )

    # Output Summation Node x^{(L)}_1 = S^k(x)
    c_out = Circle(
        (x_out, y_L),
        r_out,
        facecolor=ACTIVE_FILL,
        edgecolor=ACTIVE_DARK,
        linewidth=2.4,
        zorder=6,
    )
    ax.add_patch(c_out)
    ax.text(
        x_out,
        y_L,
        r"$\bm{\Sigma}$",
        ha="center",
        va="center",
        fontsize=15.5,
        color=ACTIVE_DARK,
        zorder=7,
    )

    # Output arrow from top of output node to the output equation banner
    y_eq_box = y_L + 1.15
    ax.annotate(
        "",
        xy=(x_out, y_eq_box - 0.38),
        xytext=(x_out, y_L + r_out + 0.04),
        arrowprops=dict(arrowstyle="-|>", color=ACTIVE_DARK, lw=1.8, mutation_scale=11),
        zorder=5,
    )

    # Top Output Equation Banner
    ax.text(
        x_out,
        y_eq_box,
        r"$S^k(x) = x^{(L)}_1 = (\bm{\Phi}_L \circ \cdots \circ \bm{\Phi}_2 \circ \bm{\Phi}_1)(x), \qquad \partial_k S^k(x) = \textstyle\sum_{\mathcal{P} \in \mathbb{P}_k} \prod_{l=1}^L \phi'_{l, i_l, i_{l-1}} \geq \prod_{l=1}^L \epsilon_l = \epsilon > 0$",
        ha="center",
        va="center",
        fontsize=10.8,
        color=SLATE_DARK,
        bbox=dict(boxstyle="round,pad=0.40", facecolor="#f8fafc", edgecolor=ACTIVE_MAIN, lw=1.35),
        zorder=7,
    )

    # -------------------------------------------------------------------------
    # 5. SYMMETRIC 2x2 MARGIN CALLOUTS (Theorem 4.8 Conditions)
    # -------------------------------------------------------------------------
    left_box_x = -2.15

    # (a) Condition 1: Target Input Connections (l = 0, i = k)
    ax.text(
        left_box_x,
        y_box0,
        r"\textbf{Target Input} ($l=1$)" + "\n"
        + r"$\phi'_{1,j,k}(x_k) \geq 0, \; \phi'_{1,i^*_1,k} \geq \epsilon_1 > 0$",
        ha="center",
        va="center",
        linespacing=1.45,
        fontsize=10.2,
        color=ACTIVE_DARK,
        bbox=dict(boxstyle="round,pad=0.44", facecolor=ACTIVE_BOX_FILL, edgecolor=ACTIVE_MAIN, lw=1.25),
        zorder=7,
    )
    ax.annotate(
        "",
        xy=(x_inputs[0] - dx_box0 - 0.5 * s0 - 0.10, y_box0),
        xytext=(-0.58, y_box0),
        arrowprops=dict(arrowstyle="->", color=ACTIVE_MAIN, lw=1.35),
        zorder=6,
    )

    # (b) Condition 2a: Certified Active Path P^*
    ax.text(
        left_box_x,
        y_box1,
        r"\textbf{Active Path} $\mathcal{P}^* \in \mathbb{P}_k$" + "\n"
        + r"$\phi'_{l, i^*_l, i^*_{l-1}} \geq \epsilon_l > 0$",
        ha="center",
        va="center",
        linespacing=1.45,
        fontsize=10.2,
        color=ACTIVE_DARK,
        bbox=dict(boxstyle="round,pad=0.44", facecolor=ACTIVE_BOX_FILL, edgecolor=ACTIVE_MAIN, lw=1.45),
        zorder=7,
    )
    ax.annotate(
        "",
        xy=(1.55 - 0.5 * s1 - 0.10, y_box1),
        xytext=(-0.55, y_box1),
        arrowprops=dict(arrowstyle="->", color=ACTIVE_MAIN, lw=1.5),
        zorder=6,
    )

    # Right Margin: Layer Hierarchy (Liu et al. Fig 0.1d style)
    rx_label = 10.85
    ax.text(
        rx_label,
        y_L,
        r"\textbf{Output Layer} $[n_L = 1]$",
        ha="left",
        va="center",
        fontsize=10.5,
        color=SLATE_DARK,
    )
    ax.text(
        rx_label,
        y_box2,
        r"Outer Layer $\bm{\Phi}_L$",
        ha="left",
        va="center",
        fontsize=10.2,
        color=ACTIVE_DARK,
    )
    ax.text(
        rx_label,
        y_Lm1,
        r"\textbf{Hidden Layer} $\mathbf{x}^{(L-1)} \in \mathbb{R}^{n_{L-1}}$",
        ha="left",
        va="center",
        fontsize=10.5,
        color=SLATE_DARK,
    )
    ax.text(
        rx_label,
        y_1,
        r"\textbf{Hidden Layer} $\mathbf{x}^{(1)} \in \mathbb{R}^{n_1}$",
        ha="left",
        va="center",
        fontsize=10.5,
        color=SLATE_DARK,
    )
    ax.text(
        rx_label,
        y_0,
        r"\textbf{Input Layer} $[n_0 = d]$",
        ha="left",
        va="center",
        fontsize=10.5,
        color=SLATE_DARK,
    )

    # Right Margin: Condition 2b (Other Downstream Connections, 1 <= l <= L-1)
    ax.text(
        12.70,
        y_box1,
        r"\textbf{Downstream} ($2 \leq l \leq L$)" + "\n"
        + r"Non-decreasing ($\phi'_{l,j,i} \geq 0$)",
        ha="center",
        va="center",
        linespacing=1.45,
        fontsize=10.2,
        color=MONO_SOFT_DARK,
        bbox=dict(boxstyle="round,pad=0.44", facecolor=MONO_SOFT_BOX, edgecolor=MONO_SOFT_MAIN, lw=1.25),
        zorder=7,
    )
    ax.annotate(
        "",
        xy=(9.05 + 0.5 * s1 + 0.10, y_box1),
        xytext=(10.90, y_box1),
        arrowprops=dict(arrowstyle="->", color=MONO_SOFT_MAIN, lw=1.35),
        zorder=6,
    )

    # Right Margin: Condition 3 (Conditioning Input Connections, l = 0, i != k)
    ax.text(
        12.70,
        y_box0,
        r"\textbf{Conditioning} ($l=1, i \neq k$)" + "\n"
        + r"Unconstrained ($\phi'_{1,j,i} \in \mathbb{R}$)",
        ha="center",
        va="center",
        linespacing=1.45,
        fontsize=10.2,
        color=BLUE_DARK,
        bbox=dict(boxstyle="round,pad=0.44", facecolor=BLUE_BOX_FILL, edgecolor=BLUE_MAIN, lw=1.25),
        zorder=7,
    )
    ax.annotate(
        "",
        xy=(x_inputs[2] + dx_box0 + 0.5 * s0 + 0.10, y_box0),
        xytext=(11.12, y_box0),
        arrowprops=dict(arrowstyle="->", color=BLUE_MAIN, lw=1.35),
        zorder=6,
    )

    pdf_path = os.path.join(out_dir, "fig_deep_kan_liu.pdf")
    png_path = os.path.join(out_dir, "fig_deep_kan_liu.png")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("Successfully generated:", pdf_path, png_path)


if __name__ == "__main__":
    generate_deep_kan_liu_style()
