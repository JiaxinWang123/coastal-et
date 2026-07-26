"""Data-flow schematic (conceptual, no code): data products and the processing steps.

Main path is bold; flux-footprint weighting is a robustness test (not in production).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

R = "/anvil/scratch/x-jwang120/coastal-et"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"]})
SRC, DAT, TAB, MOD, OUT = "#4C72B0", "#55A868", "#8172B3", "#DD8452", "#C44E52"
INK = "#3a3a3a"

fig = plt.figure(figsize=(15.5, 8.3)); fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
ax.text(2, 96.8, "Coastal-wetland ET — data-flow schematic", fontsize=20, fontweight="bold", color="#111")
ax.text(2, 92.4, "boxes = data products · bold arrows = the processing step",
        fontsize=12, color="#555")

for x, num, lab, c in [(9, "1", "SOURCES", SRC), (33, "2", "INTERIM / PROCESSED", DAT),
                       (57, "3", "ANALYSIS TABLE", TAB), (75.5, "4", "MODEL", MOD), (92, "5", "OUTPUT", OUT)]:
    ax.text(x, 87, f"{num}   {lab}", fontsize=11.5, fontweight="bold", color=c, ha="center")


def node(x, y, w, h, header, lines, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.9",
                                fc="white", ec=c, lw=1.9, zorder=3))
    ax.add_patch(FancyBboxPatch((x, y + h - 3.3), w, 3.3, boxstyle="round,pad=0,rounding_size=0.4",
                                fc=c, ec="none", alpha=0.92, zorder=4))
    ax.text(x + w / 2, y + h - 1.65, header, fontsize=11.5, color="white", ha="center", va="center",
            fontweight="bold", zorder=5)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 6.0 - i * 3.5, ln, fontsize=10, color="#222", ha="center", va="center", zorder=5)
    return (x, y, w, h)


def arrow(a, b, label="", col=INK, lw=2.4, ly=1.8, fs=9.4):
    ax1, ay = a[0] + a[2], a[1] + a[3] / 2
    bx, by = b[0], b[1] + b[3] / 2
    ax.add_patch(FancyArrowPatch((ax1, ay), (bx, by), arrowstyle="-|>", mutation_scale=19,
                                 color=col, lw=lw, zorder=2))
    if label:
        ax.text((ax1 + bx) / 2, (ay + by) / 2 + ly, label, fontsize=fs, color=INK,
                ha="center", va="bottom", zorder=6, fontweight="bold")


# ---- MAIN PATH ----
s1 = node(1.5, 68, 16, 14, "flux towers", ["AmeriFlux /", "FLUXNET"], SRC)
s2 = node(1.5, 46, 16, 14, "satellite", ["Landsat +", "Sentinel-2"], SRC)
s3 = node(1.5, 24, 16, 14, "meteorology", ["gridMET / ERA5"], SRC)
i1 = node(24, 68, 22, 14, "daily ET", ["energy-balance", "closed  (target)"], DAT)
i2 = node(24, 46, 22, 14, "satellite indices", ["7 indices", "500 m window"], DAT)
i3 = node(24, 24, 22, 14, "meteorology", ["TA, VPD, SW, WS", "+ reference ET"], DAT)
t = node(51, 46, 17, 14, "feature table", ["833 site-days", "13 sites × 14"], TAB)
m = node(71, 46, 13, 14, "model", ["ExtraTrees", "11 features"], MOD)
o = node(87, 46, 12, 14, "30 m ET maps", ["per reserve"], OUT)
shp = node(71, 26, 13, 8, "target polygons", ["7 NERR reserves"], "#8a8a8a")

arrow(s1, i1, "energy-balance\nclosure")
arrow(s2, i2, "cloud mask +\nper-pixel indices")
arrow(s3, i3, "extract + FAO-56 ETo")
for src_node in (i1, i2, i3):
    arrow(src_node, t, "", col=INK)
ax.text(48.6, 63.8, "match cloud-free\noverpass → features + ET", fontsize=9.2, color=INK,
        ha="center", fontweight="bold")
arrow(t, m, "", col=INK)
ax.text(69.5, 63.8, "feature selection\n+ train (leave-site CV)", fontsize=9.2, color=INK,
        ha="center", fontweight="bold")
arrow(m, o, "", col=INK)
ax.text(85.5, 63.8, "predict per pixel,\nmask water, clip", fontsize=9.2, color=INK,
        ha="center", fontweight="bold")
ax.add_patch(FancyArrowPatch((84, 32), (91, 46), arrowstyle="-|>", mutation_scale=16, color="#9a9a9a", lw=1.9, zorder=2))

# ---- ROBUSTNESS TEST band ----
by0, bh = 4, 14
ax.add_patch(FancyBboxPatch((1.5, by0), 82, bh, boxstyle="round,pad=0.3,rounding_size=1.0",
                            fc="#f3f3f3", ec="#c9c9c9", lw=1.2, ls="--", zorder=1))
ax.text(3.6, by0 + bh - 3.2, "ROBUSTNESS TEST — flux footprint (not used in production)",
        fontsize=11, fontweight="bold", color="#666")
ax.text(3.6, by0 + bh - 7.6,
        "Kljun (2015) 2-D footprint computed for all 13 towers; satellite features footprint-weighted "
        "instead of window-averaged.", fontsize=9.8, color="#555")
ax.text(3.6, by0 + 2.8,
        "Result: leave-site R² = 0.71 ≈ 0.72 (500 m window) → the model is support-invariant; "
        "the 500 m window is kept.", fontsize=10, color="#8f342f", fontweight="bold")

ax.text(85.5, 16, "Anvil HPC · Python 3.11\nreproducible notebooks 00–04",
        fontsize=9.2, color="#777", ha="left", va="top")
ax.text(85.5, 8.5, "Validation: leave-site-out\nR² ≈ 0.72", fontsize=10.5, color="#8f342f",
        fontweight="bold", ha="left", va="top")

fig.savefig(f"{R}/figures/dataflow_diagram.png", dpi=200, facecolor="white", bbox_inches="tight")
print("wrote figures/dataflow_diagram.png")
