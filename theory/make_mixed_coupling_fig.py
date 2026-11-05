import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Publication typography
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
})

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(out_dir, exist_ok=True)

fig = plt.figure(figsize=(13.5, 4.4), dpi=300)
gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.0, 1.1], wspace=0.32)

# =============================================================================
# Panel (a): 6D Graphical Model Architecture
# =============================================================================
ax0 = fig.add_subplot(gs[0])
ax0.set_xlim(-0.2, 5.2)
ax0.set_ylim(-0.3, 3.8)
ax0.axis("off")
ax0.set_title(r"\textbf{(a) 6D Mixed Coupling Benchmark Graph}", pad=12, loc="left", weight="bold")

# Node positions (x, y)
# 1-indexed notation in text: X_1, X_2, X_3, X_4, X_5, X_6
pos = {
    1: (0.7, 3.0),
    2: (0.7, 1.0),
    3: (2.7, 3.0),
    4: (2.7, 1.0),
    5: (4.4, 3.0),
    6: (4.4, 1.0),
}

# Subgraph bounding boxes
# Regime 1: Additive pair (1, 2)
box1 = patches.FancyBboxPatch((-0.05, 0.4), 1.5, 3.1, boxstyle="round,pad=0.15",
                             facecolor="#EBF3FB", edgecolor="#2B6CB0", linestyle="--", linewidth=1.5, alpha=0.6)
ax0.add_patch(box1)
ax0.text(0.7, 3.65, "Pure Additive\n$X_2 = 1.2\\sin(X_1) + Z_2$", ha="center", va="center", fontsize=8.5,
         color="#1A365D", weight="bold")

# Regime 2: Multiplicative pair (3, 4)
box2 = patches.FancyBboxPatch((1.95, 0.4), 1.5, 3.1, boxstyle="round,pad=0.15",
                             facecolor="#FEF3C7", edgecolor="#D97706", linestyle="--", linewidth=1.5, alpha=0.6)
ax0.add_patch(box2)
ax0.text(2.7, 3.65, "Pure Multiplicative\n$X_4 = X_3 \\cdot Z_4$", ha="center", va="center", fontsize=8.5,
         color="#78350F", weight="bold")

# Regime 3: Mixed node X_6
box3 = patches.FancyBboxPatch((3.75, 0.4), 1.45, 3.1, boxstyle="round,pad=0.15",
                             facecolor="#F3E8FF", edgecolor="#7C3AED", linestyle="--", linewidth=1.5, alpha=0.6)
ax0.add_patch(box3)
ax0.text(4.47, 3.65, "Mixed Node $X_6$\n$1.2 X_5 + X_3 \\cdot Z_6$", ha="center", va="center", fontsize=8.5,
         color="#4C1D95", weight="bold")

# Edges
# (1, 2) Additive
ax0.annotate("", xy=pos[2], xytext=pos[1],
             arrowprops=dict(arrowstyle="-", color="#2B6CB0", lw=3.0))
ax0.text(0.95, 2.0, "Additive\nShift", color="#2B6CB0", fontsize=8, ha="left", va="center", weight="bold")

# (3, 4) Multiplicative
ax0.annotate("", xy=pos[4], xytext=pos[3],
             arrowprops=dict(arrowstyle="-", color="#D97706", lw=3.0))
ax0.text(2.95, 2.0, "Scale\nMod.", color="#D97706", fontsize=8, ha="left", va="center", weight="bold")

# (5, 6) Additive edge
ax0.annotate("", xy=pos[6], xytext=pos[5],
             arrowprops=dict(arrowstyle="-", color="#7C3AED", lw=3.0))
ax0.text(4.7, 2.0, "Additive\nShift", color="#7C3AED", fontsize=8, ha="left", va="center", weight="bold")

# (3, 6) Multiplicative cross-edge
ax0.annotate("", xy=pos[6], xytext=pos[3],
             arrowprops=dict(arrowstyle="-", color="#D97706", lw=2.2, linestyle="-."))
ax0.text(3.4, 1.8, "Scale\nMod.", color="#D97706", fontsize=7.5, ha="center", va="center", weight="bold",
         bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#D97706", lw=0.8))

# Draw nodes
node_colors = {
    1: "#93C5FD", 2: "#3B82F6",
    3: "#FCD34D", 4: "#F59E0B",
    5: "#C4B5FD", 6: "#8B5CF6",
}
for n, p in pos.items():
    circle = plt.Circle(p, 0.28, facecolor=node_colors[n], edgecolor="#1E293B", lw=1.8, zorder=5)
    ax0.add_patch(circle)
    ax0.text(p[0], p[1], f"$X_{n}$", ha="center", va="center", fontsize=10.5, weight="bold",
             color="#0F172A", zorder=6)

# =============================================================================
# Panel (b): Recovered Precision Matrix Heatmap (SM-KAN)
# =============================================================================
ax1 = fig.add_subplot(gs[1])
ax1.set_title(r"\textbf{(b) Recovered Precision } $\widehat{\Omega}_{\text{SM-KAN}}$", pad=12, loc="left", weight="bold")

raw_omega = np.array([
    [1.000, 1.007, 0.095, 0.177, 0.192, 0.159],
    [1.007, 1.000, 0.087, 0.156, 0.130, 0.131],
    [0.095, 0.087, 1.000, 0.403, 0.323, 0.540],
    [0.177, 0.156, 0.403, 1.000, 0.285, 0.460],
    [0.192, 0.130, 0.323, 0.285, 1.000, 3.781],
    [0.159, 0.131, 0.540, 0.460, 3.781, 1.000],
])

im = ax1.imshow(raw_omega, cmap="YlGnBu", vmin=0, vmax=1.5)
cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=8)
cbar.set_label(r"Generalized Precision $\widehat{\Omega}_{jk}$", fontsize=8.5)

labels = [f"$X_{i}$" for i in range(1, 7)]
ax1.set_xticks(range(6))
ax1.set_yticks(range(6))
ax1.set_xticklabels(labels, fontsize=9)
ax1.set_yticklabels(labels, fontsize=9)

# Highlight true edges with red boxes
true_edges_idx = [(0, 1), (1, 0), (2, 3), (3, 2), (4, 5), (5, 4), (2, 5), (5, 2)]
for r in range(6):
    for c in range(6):
        val = raw_omega[r, c]
        if (r, c) in true_edges_idx:
            rect = patches.Rectangle((c - 0.48, r - 0.48), 0.96, 0.96,
                                     linewidth=2.0, edgecolor="#DC2626", facecolor="none", zorder=4)
            ax1.add_patch(rect)
            ax1.text(c, r, f"{val:.2f}", ha="center", va="center",
                     color="#DC2626" if val < 1.0 else "white", fontsize=8, weight="bold")
        else:
            ax1.text(c, r, f"{val:.2f}", ha="center", va="center",
                     color="white" if val > 0.9 else "#334155", fontsize=7.5)

# =============================================================================
# Panel (c): Pathway Decoupling Standard Deviations
# =============================================================================
ax2 = fig.add_subplot(gs[2])
ax2.set_title(r"\textbf{(c) Pathway Decoupling Activations}", pad=12, loc="left", weight="bold")

# Target nodes to showcase
cases = [
    r"Node $X_2$ (Add)" + "\nParent: $X_1$",
    r"Node $X_4$ (Mult)" + "\nParent: $X_3$",
    r"Node $X_6$ (Mixed)" + "\nAdd Parent: $X_5$",
    r"Node $X_6$ (Mixed)" + "\nMult Parent: $X_3$",
]
psi_scale = [0.031, 0.165, 0.024, 0.181]  # Scale std(psi)
rho_offset = [0.137, 0.014, 0.161, 0.063]  # Offset std(rho)

y = np.arange(len(cases))
height = 0.35

rects1 = ax2.barh(y + height/2, psi_scale, height, label=r"Scale Spline $\operatorname{std}(\psi_j)$",
                  color="#F59E0B", edgecolor="#B45309", lw=1.2)
rects2 = ax2.barh(y - height/2, rho_offset, height, label=r"Offset Spline $\operatorname{std}(\rho_j)$",
                  color="#3B82F6", edgecolor="#1D4ED8", lw=1.2)

ax2.set_yticks(y)
ax2.set_yticklabels(cases, fontsize=8.5)
ax2.invert_yaxis()  # top-down
ax2.set_xlabel("Empirical Standard Deviation", fontsize=9)
ax2.set_xlim(0, 0.22)
ax2.grid(axis="x", linestyle=":", alpha=0.6)
ax2.legend(loc="lower right", framealpha=0.9, fontsize=8)

# Value annotations
for rect in rects1:
    w = rect.get_width()
    ax2.text(w + 0.005, rect.get_y() + rect.get_height()/2, f"{w:.3f}",
             va="center", ha="left", fontsize=7.5, color="#B45309", weight="bold")

for rect in rects2:
    w = rect.get_width()
    ax2.text(w + 0.005, rect.get_y() + rect.get_height()/2, f"{w:.3f}",
             va="center", ha="left", fontsize=7.5, color="#1D4ED8", weight="bold")

plt.tight_layout()
pdf_path = os.path.join(out_dir, "fig6_mixed_coupling.pdf")
png_path = os.path.join(out_dir, "fig6_mixed_coupling.png")
plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(png_path, bbox_inches="tight", dpi=300)
print(f"Generated {pdf_path} and {png_path}")
