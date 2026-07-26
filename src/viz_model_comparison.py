"""Nature-style model comparison figure: R2 across 3 prediction scenarios."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/anvil/scratch/x-jwang120/coastal-et"
t = pd.read_csv(f"{R}/data/processed/model_comparison.csv")
t = t.sort_values("leave_year", ascending=True).reset_index(drop=True)  # best at top

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.linewidth": 0.7, "xtick.major.width": 0.7,
    "ytick.major.width": 0.7, "xtick.major.size": 3, "ytick.major.size": 3,
})
FAMCOL = {"GP": "#B8860B", "linear": "#4C72B0", "trees": "#55A868",
          "kernel": "#DD8452", "deep": "#C44E52", "other": "#999999"}
INK, GREY = "#1a1a1a", "#8a8a8a"
COLS = [("random_CV", "a", "Random K-fold", "monitored sites"),
        ("leave_year", "b", "Leave-year-out", "unseen years"),
        ("leave_tower", "c", "Leave-tower-out", "unseen site")]
XMIN = -1.15   # clip; off-scale values annotated

fig, axes = plt.subplots(1, 3, figsize=(7.4, 4.2), sharey=True, constrained_layout=True)
fig.patch.set_facecolor("white")
y = np.arange(len(t))
for ax, (col, tag, title, sub) in zip(axes, COLS):
    ax.set_facecolor("white")
    vals = t[col].clip(lower=XMIN)
    cols = [FAMCOL.get(f, "#999") for f in t.family]
    ax.barh(y, vals, color=cols, height=0.7, edgecolor="white", linewidth=0.4, zorder=3)
    ax.axvline(0, color=GREY, lw=0.8, zorder=2)
    # annotate off-scale (heavily negative) values
    for yi, v in zip(y, t[col]):
        if v < XMIN:
            ax.text(XMIN + 0.02, yi, f"{v:.1f}", va="center", ha="left",
                    fontsize=6, color="white", zorder=4)
    # mark the best in each scenario
    bi = int(np.argmax(t[col].values))
    ax.text(t[col].iloc[bi] + 0.03 if t[col].iloc[bi] >= 0 else 0.03, bi,
            f"{t[col].iloc[bi]:.2f}", va="center", ha="left", fontsize=6.5,
            fontweight="bold", color=INK, zorder=4)
    ax.set_xlim(XMIN, 0.75)
    ax.set_xticks([-1, -0.5, 0, 0.5])
    ax.set_title(title, fontsize=8.5, color=INK, pad=3)
    ax.text(0.5, 1.055, f"predict at {sub}", transform=ax.transAxes, ha="center",
            fontsize=6.5, color=GREY, style="italic")
    ax.text(-0.02, 1.09, tag, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="right", color=INK)
    ax.set_xlabel("$R^2$", fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=7, colors=INK)
axes[0].set_yticks(y)
axes[0].set_yticklabels(t.model, fontsize=7)
# family legend
from matplotlib.patches import Patch
leg = [Patch(fc=FAMCOL[k], label=v) for k, v in
       [("GP", "Gaussian process"), ("linear", "linear"), ("trees", "tree ensembles"),
        ("kernel", "kernel / kNN"), ("deep", "deep learning")]]
axes[2].legend(handles=leg, frameon=False, fontsize=6.5, loc="lower right",
               handlelength=1.1, handletextpad=0.4, labelspacing=0.3)
fig.suptitle("Model comparison — predictive ET skill (footprint features + ERA5, n=227)",
             fontsize=10, fontweight="bold", color=INK)
for ext in ("png", "pdf"):
    fig.savefig(f"{R}/figures/model_comparison.{ext}", dpi=450, facecolor="white",
                bbox_inches="tight")
print("wrote model_comparison.png (+ .pdf)")
