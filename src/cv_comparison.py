"""Figure: predictive skill across the three validation schemes, 5 vs 13 sites.

K-fold (predict at monitored sites) | leave-year-out (unseen years) |
leave-site-out (scale to an unseen tower). The three schemes answer progressively
harder questions; the 5-site vs 13-site contrast shows site diversity is what makes
leave-site-out (true upscaling) possible.
"""
import sys
sys.path.insert(0, "/anvil/scratch/x-jwang120/coastal-et/src")
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import r2_score
import more_sites as M
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = pd.read_parquet("/anvil/scratch/x-jwang120/coastal-et/data/processed/more_sites_table.parquet")
FE = M.FEATS
if "year" not in d.columns:                       # reconstruct if absent
    d["year"] = pd.to_datetime(d.get("date"), utc=True, errors="coerce").dt.year
print("year coverage:", int(d.year.notna().sum()), "/", len(d))


def models():
    return {"ExtraTrees": ExtraTreesRegressor(400, min_samples_leaf=3, random_state=0, n_jobs=-1),
            "GaussProc": make_pipeline(StandardScaler(), GaussianProcessRegressor(
                kernel=ConstantKernel() * RBF() + WhiteKernel(), normalize_y=True,
                alpha=1e-3, random_state=0))}


def kfold(data, mk):
    yt, yp = [], []
    for tri, tei in KFold(10, shuffle=True, random_state=0).split(data):
        m = clone(mk).fit(data.iloc[tri][FE].values, data.iloc[tri].ET_closed_mm.values)
        yp.append(m.predict(data.iloc[tei][FE].values)); yt.append(data.iloc[tei].ET_closed_mm.values)
    return r2_score(np.concatenate(yt), np.concatenate(yp))


def leave_year(data, sites, mk):
    yt, yp = [], []
    for s in sites:
        ds = data[data.SITE_ID == s]
        for y in sorted(ds.year.dropna().unique()):
            tr, te = ds[ds.year != y], ds[ds.year == y]
            if len(te) < 5 or len(tr) < 15:
                continue
            m = clone(mk).fit(tr[FE].values, tr.ET_closed_mm.values)
            yp.append(m.predict(te[FE].values)); yt.append(te.ET_closed_mm.values)
    return r2_score(np.concatenate(yt), np.concatenate(yp))


def leave_site(data, sites, mk):
    yt, yp = [], []
    for s in sites:
        tr, te = data[data.SITE_ID != s], data[data.SITE_ID == s]
        if len(te) < 5:
            continue
        m = clone(mk).fit(tr[FE].values, tr.ET_closed_mm.values)
        yp.append(m.predict(te[FE].values)); yt.append(te.ET_closed_mm.values)
    return r2_score(np.concatenate(yt), np.concatenate(yp))


ev = [s for s in M.EVERGLADES if s in d.SITE_ID.unique()]
alls = sorted(d.SITE_ID.unique())
d5, d13 = d[d.SITE_ID.isin(ev)], d
res = {}
for mn, mk in models().items():
    for label, data, sites in [("5 sites", d5, ev), ("13 sites", d13, alls)]:
        res[(mn, label)] = [kfold(data, mk), leave_year(data, sites, mk), leave_site(data, sites, mk)]
        print(f"{mn:<11}{label:<9} K-fold={res[(mn,label)][0]:.3f}  "
              f"leave-year={res[(mn,label)][1]:.3f}  leave-site={res[(mn,label)][2]:.3f}")

# ---- Nature-style grouped bar figure ----
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 3, "ytick.major.size": 3})
INK, GREY = "#1a1a1a", "#8a8a8a"
schemes = ["K-fold\n(monitored\nsites)", "Leave-year-out\n(unseen\nyears)", "Leave-site-out\n(unseen\ntower)"]
C5, C13 = "#B0C4DE", "#2C5F8A"          # light = 5 sites, dark = 13 sites
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), sharey=True, constrained_layout=True)
fig.patch.set_facecolor("white")
x = np.arange(3); w = 0.36
for ax, mn in zip(axes, ["ExtraTrees", "GaussProc"]):
    ax.set_facecolor("white")
    v5, v13 = res[(mn, "5 sites")], res[(mn, "13 sites")]
    b1 = ax.bar(x - w/2, np.clip(v5, -1.05, 1), w, color=C5, label="5 Everglades", zorder=3)
    b2 = ax.bar(x + w/2, np.clip(v13, -1.05, 1), w, color=C13, label="13 wetlands", zorder=3)
    ax.axhline(0, color=GREY, lw=0.8, zorder=2)
    for bars, vals in [(b1, v5), (b2, v13)]:
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, (val if val >= 0 else -1.05) + (0.03 if val >= 0 else -0.0),
                    f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top",
                    fontsize=6.3, color=INK, zorder=4)
    ax.set_title(mn, fontsize=9, color=INK, pad=4)
    ax.set_xticks(x); ax.set_xticklabels(schemes, fontsize=6.8)
    ax.set_ylim(-1.15, 1.0); ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    if ax is axes[0]:
        ax.set_ylabel("$R^2$", fontsize=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK, labelsize=7)
axes[0].text(-0.14, 1.06, "a", transform=axes[0].transAxes, fontsize=11, fontweight="bold", va="top")
axes[1].text(-0.08, 1.06, "b", transform=axes[1].transAxes, fontsize=11, fontweight="bold", va="top")
axes[1].legend(frameon=False, fontsize=7, loc="upper right")
fig.suptitle("Predictive skill by validation scheme — harder tests need more sites",
             fontsize=10.5, fontweight="bold", color=INK)
for ext in ("png", "pdf"):
    fig.savefig(f"/anvil/scratch/x-jwang120/coastal-et/figures/cv_comparison.{ext}",
                dpi=450, facecolor="white", bbox_inches="tight")
print("wrote cv_comparison.png (+ .pdf)")
