"""Predictive ET model: ET = f(LST, LAI, EVI2, NDVI, NDWI, MNDWI, ERA5 met).

NOT gap-filling. This trains a regression to PREDICT ET from the observed
satellite + meteorology on real overpass days, and evaluates it three ways --
each answers a different 'predict onto what?' question:

  A. RANDOM K-FOLD (pooled)   predict held-out site-dates, model has seen all
                              sites -> predictive skill at MONITORED sites
  B. LEAVE-YEAR-OUT           predict unseen years at known sites -> temporal
  C. LEAVE-TOWER-OUT          predict an UNSEEN site -> spatial upscaling

Features are footprint-weighted per overpass date (Kljun 2015). Target is the
measured, energy-balance-closed tower ET on that date.
"""
import os
import sys
sys.path.insert(0, "/anvil/scratch/x-jwang120/coastal-et/src")
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import train_indices_model as T   # reuse its footprint-weighted build()

R = "/anvil/scratch/x-jwang120/coastal-et"
FIG = f"{R}/figures"
OUT = f"{R}/data/processed"
SITES = ["US-Esm", "US-TaS", "US-Skr", "US-Elm", "US-EvM"]
FEATS = T.FEATS   # LAI,EVI2,SAVI,NDVI,NDWI,MNDWI,LST_K + TA,VPD,SW,WS,ETo,DOYsin,DOYcos


def models():
    return {
        "GaussianProcess": lambda: make_pipeline(StandardScaler(),
            GaussianProcessRegressor(kernel=ConstantKernel() * RBF() + WhiteKernel(),
                                     normalize_y=True, alpha=1e-3, random_state=0)),
        "RandomForest": lambda: RandomForestRegressor(400, min_samples_leaf=3,
                                                      random_state=42, n_jobs=-1),
        "GradBoost": lambda: GradientBoostingRegressor(n_estimators=300, max_depth=2, random_state=42),
        "Ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10)),
    }


def oof_random(d, mk, k=10):
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    pr = []
    for tri, tei in kf.split(d):
        tr, te = d.iloc[tri], d.iloc[tei]
        m = mk()
        m.fit(tr[FEATS].values, tr.ET_closed_mm.values)
        pr.append(te.assign(pred=m.predict(te[FEATS].values)))
    return pd.concat(pr)


def oof_year(d, mk):
    pr = []
    for s in SITES:
        ds = d[d.SITE_ID == s]
        for y in sorted(ds.year.unique()):
            tr, te = ds[ds.year != y], ds[ds.year == y]
            if len(te) < 5 or len(tr) < 20:
                continue
            m = mk()
            m.fit(tr[FEATS].values, tr.ET_closed_mm.values)
            pr.append(te.assign(pred=m.predict(te[FEATS].values)))
    return pd.concat(pr)


def oof_tower(d, mk):
    pr = []
    for s in SITES:
        tr, te = d[d.SITE_ID != s], d[d.SITE_ID == s]
        if len(te) < 5:
            continue
        m = mk()
        m.fit(tr[FEATS].values, tr.ET_closed_mm.values)
        pr.append(te.assign(pred=m.predict(te[FEATS].values)))
    return pd.concat(pr)


def sc(g):
    return (r2_score(g.ET_closed_mm, g.pred),
            np.sqrt(mean_squared_error(g.ET_closed_mm, g.pred)),
            mean_absolute_error(g.ET_closed_mm, g.pred))


def main():
    d = T.build().dropna(subset=["ET_closed_mm"] + FEATS).reset_index(drop=True)
    print(f"{len(d)} tower-date samples (real overpass), {d.SITE_ID.nunique()} sites\n")
    print(f"{'model':<14}{'random-CV':>11}{'leave-year':>12}{'leave-tower':>13}")
    print("-" * 50)
    results = {}
    for name, mk in models().items():
        gr, gy, gt = oof_random(d, mk), oof_year(d, mk), oof_tower(d, mk)
        results[name] = (gr, gy, gt)
        print(f"{name:<14}{sc(gr)[0]:>11.3f}{sc(gy)[0]:>12.3f}{sc(gt)[0]:>13.3f}")

    rf = models()["RandomForest"]()
    rf.fit(d[FEATS].values, d.ET_closed_mm.values)
    imp = pd.Series(rf.feature_importances_, index=FEATS).sort_values(ascending=False)
    print("\nRF importance:", ", ".join(f"{k}={v:.2f}" for k, v in imp.head(8).items()))

    # ---- Nature-style predicted-vs-observed scatter ----
    best = max(results, key=lambda n: sc(results[n][0])[0])
    gr, gy, gt = results[best]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.linewidth": 0.7, "xtick.major.width": 0.7,
        "ytick.major.width": 0.7, "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.direction": "out", "ytick.direction": "out",
        "svg.fonttype": "none",
    })
    COL = {"US-Esm": "#4C72B0", "US-TaS": "#DD8452", "US-Skr": "#55A868",
           "US-Elm": "#8172B3", "US-EvM": "#64B5CD"}  # muted, cvd-safe
    INK, GREY = "#1a1a1a", "#8a8a8a"
    panels = [("a", gr, "Random K-fold", "predict at monitored sites"),
              ("b", gy, "Leave-year-out", "predict unseen years"),
              ("c", gt, "Leave-tower-out", "scale to an unseen site")]
    LIM = (0, 8)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), constrained_layout=True)
    fig.patch.set_facecolor("white")
    for ax, (tag, g, title, sub) in zip(axes, panels):
        ax.set_facecolor("white")
        ax.set_aspect("equal")
        # 1:1 line
        ax.plot(LIM, LIM, ls=(0, (4, 3)), color=GREY, lw=0.8, zorder=1)
        for s in SITES:
            gg = g[g.SITE_ID == s]
            if len(gg):
                ax.scatter(gg.ET_closed_mm, gg.pred, s=9, color=COL[s],
                           alpha=0.7, edgecolors="white", linewidths=0.2,
                           zorder=3, label=s.replace("US-", ""))
        # OLS fit (predicted on observed)
        x, y = g.ET_closed_mm.values, g.pred.values
        b1, b0 = np.polyfit(x, y, 1)
        xx = np.array(LIM)
        ax.plot(xx, b0 + b1 * xx, color=INK, lw=1.1, zorder=2)
        r2, rmse, mae = sc(g)
        ax.text(0.05, 0.965, f"$R^2$ = {r2:.2f}\nRMSE = {rmse:.2f}\n"
                f"slope = {b1:.2f}\n$n$ = {len(g)}", transform=ax.transAxes,
                fontsize=7, va="top", ha="left", color=INK, linespacing=1.35)
        ax.text(0.97, 0.06, sub, transform=ax.transAxes, fontsize=6.5,
                ha="right", va="bottom", color=GREY, style="italic")
        ax.set_title(title, fontsize=8.5, color=INK, pad=3)
        ax.text(-0.16, 1.06, tag, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="top", ha="left", color=INK)
        ax.set_xlim(LIM); ax.set_ylim(LIM)
        ax.set_xticks([0, 2, 4, 6, 8]); ax.set_yticks([0, 2, 4, 6, 8])
        ax.set_xlabel("Observed ET (mm d$^{-1}$)", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("Predicted ET (mm d$^{-1}$)", fontsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=INK, labelsize=7)
    axes[0].legend(frameon=False, fontsize=6.5, loc="lower right",
                   handletextpad=0.2, labelspacing=0.25, borderpad=0.2)
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIG}/et_prediction_scatter.{ext}", dpi=450,
                    facecolor="white", bbox_inches="tight")
    print(f"\nwrote {FIG}/et_prediction_scatter.png (+ .pdf)")


if __name__ == "__main__":
    main()
