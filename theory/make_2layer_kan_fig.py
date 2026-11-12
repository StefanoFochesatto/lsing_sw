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

    # Subtle drop shadow (stronger on active path q^*)
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

    # Subtle spline boundary dots
    ax.scatter(
        [px[0], px[-1]],
        [py[0], py[-1]],
        s=12 if highlight_active else 9,
        color=curve_color,
        zorder=6.2,
    )


def generate_2layer_kan_liu_style():
    fig, ax = plt.subplots(figsize=(12.6, 7.2), dpi=300)
    ax.set_aspect("equal")
    ax.set_xlim(-3.75, 14.15)
    ax.set_ylim(-0.65, 9.75)
    ax.axis("off")

    # Active path q^* (bold certified strictly increasing path)
    ACTIVE_DARK = "#0f5132"
    ACTIVE_MAIN = "#15803d"
    ACTIVE_FILL = "#dcfce7"
    ACTIVE_BOX_FILL = "#f0fdf4"

    # Non-decreasing monotonic paths (\phi' >= 0, \Phi' >= 0)
    MONO_SOFT_DARK = "#2d6a4f"
    MONO_SOFT_MAIN = "#52b788"
    MONO_SOFT_LINE = "#68b092"
    MONO_SOFT_FILL = "#f0f7f3"
    MONO_SOFT_BOX = "#f7fbf9"

    # Conditioning inputs (blue)
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

    # Generous vertical levels (bottom-to-top Liu et al. 2024 architecture layout)
    y_in = 0.50
    y_box1 = 2.48
    y_hid = 4.68
    y_box2 = 6.38
    y_out = 8.05

    s1 = 0.80  # Inner spline box side length
    s2 = 0.84  # Outer spline box side length
    r_in = 0.36
    r_hid = 0.36
    r_out = 0.38

    # Horizontal positions with generous breathing room between groups
    x_inputs = [1.75, 5.30, 8.85]    # Target x_k, Cond x_j, Cond x_d
    dx_box1 = 1.02
    x_hiddens = [2.15, 5.30, 8.45]   # u_{q^*}, u_q, u_Q
    x_out = 5.30

    # Horizontal level dashed guide lines (Liu et al. Fig 0.1d style)
    guide_x0, guide_x1 = 0.10, 10.60
    for y_lvl in [y_in, y_hid, y_out]:
        ax.plot(
            [guide_x0, guide_x1],
            [y_lvl, y_lvl],
            color="#cbd5e1",
            linestyle=(0, (3.5, 3.0)),
            linewidth=0.95,
            zorder=1,
        )

    # -------------------------------------------------------------------------
    # 1. INNER LAYER (Layer 1): Edges & 1D Spline Boxes \phi_{1, q, p}
    # -------------------------------------------------------------------------
    target_inner_ctrls = [
        [-2.6, -0.35, -0.08, 0.0, 0.08, 0.35, 2.6],  # Active q^*: strictly increasing spline
        [-2.4, -2.15, -1.65, 0.0, 1.65, 2.15, 2.4],  # Steep-center non-decreasing spline
        [-2.1, -1.9, -1.55, -1.0, -0.2, 0.9, 2.6],   # Convex-accelerating non-decreasing spline
    ]

    cond_j_inner_ctrls = [
        [1.9, 0.4, -1.3, -1.7, -1.1, 0.5, 2.0],      # U-shaped quadratic-like
        [-0.9, 1.7, 0.3, -1.7, -0.2, 1.8, -0.7],     # Multimodal oscillation
        [-1.6, -0.5, 1.5, 2.2, 1.3, -0.8, -1.5],     # Symmetric bell hump
    ]
    cond_d_inner_ctrls = [
        [1.5, -1.6, 0.9, 1.7, -1.5, -0.4, 1.6],      # Asymmetric wave
        [2.0, 1.4, 0.3, -0.9, -1.6, -0.5, 0.4],      # Decreasing with right hook
        [-1.5, 1.6, -1.3, -1.5, 1.7, -1.0, 1.2],     # Double-well W-curve
    ]

    all_inner_ctrls = [target_inner_ctrls, cond_j_inner_ctrls, cond_d_inner_ctrls]

    for i_idx, x_in_c in enumerate(x_inputs):
        is_target = i_idx == 0
        for q_idx, x_hid_c in enumerate(x_hiddens):
            is_active_edge = is_target and (q_idx == 0)
            x_box_c = x_in_c + (q_idx - 1) * dx_box1

            if is_active_edge:
                e_col = ACTIVE_MAIN
                b_col = ACTIVE_MAIN
                b_fill = ACTIVE_BOX_FILL
                c_col = ACTIVE_DARK
                lw_bot, lw_top = 2.65, 2.65
                alpha_t = 1.0
                z_e = 4.3
            elif is_target:
                e_col = MONO_SOFT_LINE
                b_col = MONO_SOFT_MAIN
                b_fill = MONO_SOFT_BOX
                c_col = MONO_SOFT_DARK
                lw_bot, lw_top = 1.80, 1.80
                alpha_t = 0.85
                z_e = 3.2
            else:
                e_col = BLUE_LINE
                b_col = BLUE_MAIN
                b_fill = BLUE_BOX_FILL
                c_col = BLUE_DARK
                lw_bot, lw_top = 1.45, 1.35
                alpha_t = 0.72
                z_e = 2.5

            # Edge segment: Input node -> bottom center of spline box
            ax.plot(
                [x_in_c, x_box_c],
                [y_in + r_in, y_box1 - 0.5 * s1],
                color=e_col,
                linewidth=lw_bot,
                alpha=alpha_t,
                solid_capstyle="round",
                zorder=z_e,
            )

            # Edge segment: Top center of spline box -> bottom of hidden node u_q
            ax.plot(
                [x_box_c, x_hid_c],
                [y_box1 + 0.5 * s1, y_hid - r_hid],
                color=e_col,
                linewidth=lw_top,
                alpha=alpha_t,
                solid_capstyle="round",
                zorder=z_e,
            )

            # Draw the 1D spline box
            draw_spline_box(
                ax,
                x_box_c,
                y_box1,
                s1,
                all_inner_ctrls[i_idx][q_idx],
                edge_color=b_col,
                fill_color=b_fill,
                curve_color=c_col,
                lw_box=1.95 if is_active_edge else (1.45 if is_target else 1.30),
                lw_curve=2.45 if is_active_edge else (2.00 if is_target else 1.90),
                is_monotonic=is_target,
                highlight_active=is_active_edge,
            )

    # -------------------------------------------------------------------------
    # 2. OUTER LAYER (Layer 2): Edges & 1D Spline Boxes \Phi_{2, q}
    # -------------------------------------------------------------------------
    outer_ctrls = [
        [-2.6, -0.9, 0.2, 1.0, 1.55, 1.95, 2.25],     # Active q^*: strictly increasing
        [-2.5, -0.3, -0.05, 0.05, 0.3, 0.65, 2.5],    # Non-decreasing inflection
        [-2.3, -2.05, -1.45, -0.1, 1.45, 2.05, 2.3],  # Non-decreasing sigmoidal
    ]
    outer_labels = [r"$\phi_{2,1,q^*}$", r"$\phi_{2,1,q}$", r"$\phi_{2,1,Q}$"]

    for q_idx, x_hid_c in enumerate(x_hiddens):
        is_act = (q_idx == 0)
        e_col = ACTIVE_MAIN if is_act else MONO_SOFT_LINE
        b_col = ACTIVE_MAIN if is_act else MONO_SOFT_MAIN
        b_fill = ACTIVE_BOX_FILL if is_act else MONO_SOFT_BOX
        c_col = ACTIVE_DARK if is_act else MONO_SOFT_DARK
        lw_e = 2.65 if is_act else 1.80
        z_e = 4.3 if is_act else 3.2

        # Vertical edge: Hidden node u_q -> bottom of outer spline box
        ax.plot(
            [x_hid_c, x_hid_c],
            [y_hid + r_hid, y_box2 - 0.5 * s2],
            color=e_col,
            linewidth=lw_e,
            solid_capstyle="round",
            zorder=z_e,
        )

        # Converging edge: Top center of outer spline box \Phi_{2,q} -> Output node S^k(x)
        ax.plot(
            [x_hid_c, x_out],
            [y_box2 + 0.5 * s2, y_out - r_out],
            color=e_col,
            linewidth=lw_e,
            solid_capstyle="round",
            zorder=z_e,
        )

        # Draw outer 1D spline box
        draw_spline_box(
            ax,
            x_hid_c,
            y_box2,
            s2,
            outer_ctrls[q_idx],
            edge_color=b_col,
            fill_color=b_fill,
            curve_color=c_col,
            lw_box=1.95 if is_act else 1.45,
            lw_curve=2.45 if is_act else 2.00,
            is_monotonic=True,
            highlight_active=is_act,
        )

        # Clean compact label next to each outer box
        ax.text(
            x_hid_c + 0.5 * s2 + 0.20,
            y_box2,
            outer_labels[q_idx],
            ha="left",
            va="center",
            fontsize=10.8,
            color=ACTIVE_DARK if is_act else MONO_SOFT_DARK,
            zorder=7,
        )

    # Symmetric ellipsis between u_q^*, u_q, u_Q in Hidden and Outer layers
    for xm_hid, xm_out in [(3.58, 4.12), (6.72, 7.28)]:
        ax.text(xm_hid, y_hid, r"$\cdots$", ha="center", va="center", fontsize=14, color=SLATE_MID, zorder=6)
        ax.text(xm_out, y_box2, r"$\cdots$", ha="center", va="center", fontsize=14, color=SLATE_MID, zorder=6)

    # -------------------------------------------------------------------------
    # 3. NODES: Input Layer, Hidden Summation Layer, Output Summation Layer
    # -------------------------------------------------------------------------
    # Input Nodes
    in_specs = [
        (x_inputs[0], r"$x_k$", ACTIVE_FILL, ACTIVE_MAIN, ACTIVE_DARK, 2.4),
        (x_inputs[1], r"$x_j$", BLUE_FILL, BLUE_MAIN, BLUE_DARK, 1.85),
        (x_inputs[2], r"$x_d$", BLUE_FILL, BLUE_MAIN, BLUE_DARK, 1.85),
    ]
    for xc, lbl, fc, ec, tc, lw in in_specs:
        c = Circle((xc, y_in), r_in, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=6)
        ax.add_patch(c)
        ax.text(xc, y_in, lbl, ha="center", va="center", fontsize=13.5, color=tc, zorder=7)

    # Ellipsis between conditioning nodes x_j and x_d
    ax.text(7.08, y_in, r"$\cdots$", ha="center", va="center", fontsize=14.0, color=BLUE_MAIN, zorder=6)

    # Concise single-line Input Captions below nodes
    ax.text(
        x_inputs[0],
        y_in - 0.65,
        r"\textbf{Target} $x_k$",
        ha="center",
        va="top",
        fontsize=10.5,
        color=ACTIVE_DARK,
    )

    # Bracket under conditioning inputs
    cond_x0, cond_x1 = 4.65, 9.50
    yb_cond = y_in - 0.48
    ax.plot([cond_x0, cond_x1], [yb_cond, yb_cond], color=BLUE_MAIN, linewidth=1.2)
    ax.plot([cond_x0, cond_x0], [yb_cond, yb_cond + 0.09], color=BLUE_MAIN, linewidth=1.2)
    ax.plot([cond_x1, cond_x1], [yb_cond, yb_cond + 0.09], color=BLUE_MAIN, linewidth=1.2)
    ax.text(
        0.5 * (cond_x0 + cond_x1),
        y_in - 0.65,
        r"\textbf{Conditioning} $\;x_{-k} = (x_j)_{j \neq k}$",
        ha="center",
        va="top",
        fontsize=10.5,
        color=BLUE_DARK,
    )

    # Hidden Summation Nodes (u_{q^*}, u_q, u_Q)
    hid_labels = [r"$x^{(1)}_{q^*}$", r"$x^{(1)}_q$", r"$x^{(1)}_Q$"]
    for q_idx, (xc, lbl) in enumerate(zip(x_hiddens, hid_labels)):
        is_act = (q_idx == 0)
        c = Circle(
            (xc, y_hid),
            r_hid,
            facecolor=ACTIVE_FILL if is_act else NODE_SUM_FILL,
            edgecolor=ACTIVE_MAIN if is_act else NODE_SUM_EDGE,
            linewidth=2.3 if is_act else 1.9,
            zorder=6,
        )
        ax.add_patch(c)
        ax.text(
            xc,
            y_hid,
            r"$\bm{\Sigma}$",
            ha="center",
            va="center",
            fontsize=14.5,
            color=ACTIVE_DARK if is_act else SLATE_DARK,
            zorder=7,
        )
        ax.text(
            xc - r_hid - 0.22,
            y_hid + 0.02,
            lbl,
            ha="right",
            va="center",
            fontsize=12.0,
            color=ACTIVE_DARK if is_act else SLATE_DARK,
            zorder=7,
        )

    # Output Summation Node S^k(x)
    c_out = Circle(
        (x_out, y_out),
        r_out,
        facecolor=ACTIVE_FILL,
        edgecolor=ACTIVE_DARK,
        linewidth=2.4,
        zorder=6,
    )
    ax.add_patch(c_out)
    ax.text(
        x_out,
        y_out,
        r"$\bm{\Sigma}$",
        ha="center",
        va="center",
        fontsize=15.5,
        color=ACTIVE_DARK,
        zorder=7,
    )

    # Output arrow from top of output node to the output equation banner
    y_eq_box = y_out + 1.25
    ax.annotate(
        "",
        xy=(x_out, y_eq_box - 0.42),
        xytext=(x_out, y_out + r_out + 0.04),
        arrowprops=dict(arrowstyle="-|>", color=ACTIVE_DARK, lw=1.8, mutation_scale=11),
        zorder=5,
    )

    # Compact, airy output equation above output node
    ax.text(
        x_out,
        y_eq_box,
        r"$S^k(x) = x^{(2)}_1 = \textstyle\sum_{q=1}^Q \phi_{2,1,q}\big(x^{(1)}_q\big), \qquad \partial_k S^k(x) \ge \phi'_{2,1,q^*}\phi'_{1,q^*,k} \ge \epsilon > 0$",
        ha="center",
        va="center",
        fontsize=11.2,
        color=SLATE_DARK,
        bbox=dict(boxstyle="round,pad=0.40", facecolor="#f8fafc", edgecolor=ACTIVE_MAIN, lw=1.35),
        zorder=7,
    )

    # -------------------------------------------------------------------------
    # 4. CONCISE, AIRY MARGIN ANNOTATIONS (Generous spacing & line leading)
    # -------------------------------------------------------------------------
    left_box_x = -2.10

    # (a) Target Inner Splines callout
    ax.text(
        left_box_x,
        y_box1,
        r"\textbf{Target Splines} $\phi_{1,q,k}$ ($l=1$)" + "\n"
        + r"Non-decreasing ($\phi'_{1,q,k} \ge 0$)" + "\n"
        + r"\textbf{Active path} $\mathcal{P}^*$: $\phi'_{1,q^*,k} \ge \epsilon_1 > 0$",
        ha="center",
        va="center",
        linespacing=1.45,
        fontsize=9.8,
        color=ACTIVE_DARK,
        bbox=dict(boxstyle="round,pad=0.44", facecolor=ACTIVE_BOX_FILL, edgecolor=ACTIVE_MAIN, lw=1.25),
        zorder=7,
    )
    ax.annotate(
        "",
        xy=(x_inputs[0] - dx_box1 - 0.5 * s1 - 0.10, y_box1),
        xytext=(-0.55, y_box1),
        arrowprops=dict(arrowstyle="->", color=ACTIVE_MAIN, lw=1.35),
        zorder=6,
    )

    # (b) Hidden Summation Nodes formula
    ax.text(
        left_box_x,
        y_hid,
        r"$x^{(1)}_q = \phi_{1,q,k}(x_k) + \textstyle\sum_{j \neq k} \phi_{1,q,j}(x_j)$",
        ha="center",
        va="center",
        fontsize=10.0,
        color=SLATE_DARK,
        bbox=dict(boxstyle="round,pad=0.42", facecolor="#f8fafc", edgecolor=SLATE_LIGHT, lw=1.15),
        zorder=7,
    )
    ax.annotate(
        "",
        xy=(x_hiddens[0] - r_hid - 0.68, y_hid),
        xytext=(-0.22, y_hid),
        arrowprops=dict(arrowstyle="->", color=SLATE_MID, lw=1.25),
        zorder=6,
    )

    # (c) Outer Splines callout
    ax.text(
        left_box_x,
        y_box2,
        r"\textbf{Outer Splines} $\phi_{2,1,q}$ ($l=2$)" + "\n"
        + r"Non-decreasing ($\phi'_{2,1,q} \ge 0$)" + "\n"
        + r"\textbf{Active path} $\mathcal{P}^*$: $\phi'_{2,1,q^*} \ge \epsilon_2 > 0$",
        ha="center",
        va="center",
        linespacing=1.45,
        fontsize=9.8,
        color=ACTIVE_DARK,
        bbox=dict(boxstyle="round,pad=0.44", facecolor=ACTIVE_BOX_FILL, edgecolor=ACTIVE_MAIN, lw=1.25),
        zorder=7,
    )
    ax.annotate(
        "",
        xy=(x_hiddens[0] - 0.5 * s2 - 0.10, y_box2),
        xytext=(-0.55, y_box2),
        arrowprops=dict(arrowstyle="->", color=ACTIVE_MAIN, lw=1.35),
        zorder=6,
    )

    # Right Margin: Concise single-line Layer Labels + 2-line Conditioning Box
    rx_label = 10.85
    ax.text(
        rx_label,
        y_out,
        r"\textbf{Output Layer} $[1]$",
        ha="left",
        va="center",
        fontsize=10.5,
        color=SLATE_DARK,
    )
    ax.text(
        rx_label,
        y_box2,
        r"Outer Layer $\bm{\Phi}_2$",
        ha="left",
        va="center",
        fontsize=10.2,
        color=ACTIVE_DARK,
    )
    ax.text(
        rx_label,
        y_hid,
        r"\textbf{Hidden Layer} $[Q]$",
        ha="left",
        va="center",
        fontsize=10.5,
        color=SLATE_DARK,
    )
    ax.text(
        rx_label,
        y_in,
        r"\textbf{Input Layer} $[d]$",
        ha="left",
        va="center",
        fontsize=10.5,
        color=SLATE_DARK,
    )

    # Right Margin: Conditioning Inner Splines callout
    ax.text(
        12.40,
        y_box1,
        r"\textbf{Conditioning} $\phi_{1,q,j}$ ($l=1$)" + "\n"
        + r"Unconstrained ($\phi'_{1,q,j} \in \mathbb{R}$)",
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
        xy=(x_inputs[2] + dx_box1 + 0.5 * s1 + 0.10, y_box1),
        xytext=(10.95, y_box1),
        arrowprops=dict(arrowstyle="->", color=BLUE_MAIN, lw=1.35),
        zorder=6,
    )

    pdf_path = os.path.join(out_dir, "fig_2layer_kan_liu.pdf")
    png_path = os.path.join(out_dir, "fig_2layer_kan_liu.png")
    plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0.10)
    plt.savefig(png_path, bbox_inches="tight", pad_inches=0.10)
    plt.close()
    print(f"Successfully generated: {pdf_path} {png_path}")


if __name__ == "__main__":
    generate_2layer_kan_liu_style()
