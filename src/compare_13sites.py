"""Full model comparison on the 13-site coastal-wetland dataset (n=833).

At 833 samples (vs 227 for the Everglades-only set) more expressive models -
gradient boosting and even deep nets - become viable. I re-run the whole model
family across the three validation schemes, with leave-SITE-out (spatial upscaling)
now the headline test since it is finally positive.
"""
import sys
sys.path.insert(0, "/anvil/scratch/x-jwang120/coastal-et/src")
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, HistGradientBoostingRegressor)
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import r2_score
import more_sites as M
import torch
import torch.nn as nn
torch.set_num_threads(8)

d = pd.read_parquet("/anvil/scratch/x-jwang120/coastal-et/data/processed/more_sites_table.parquet")
FE = M.FEATS
SITES = sorted(d.SITE_ID.unique())


def classical():
    m = {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10)),
        "ElasticNet": make_pipeline(StandardScaler(), ElasticNet(alpha=0.05, l1_ratio=0.3, max_iter=5000)),
        "PLS": make_pipeline(StandardScaler(), PLSRegression(n_components=6)),
        "kNN": make_pipeline(StandardScaler(), KNeighborsRegressor(10, weights="distance")),
        "SVR": make_pipeline(StandardScaler(), SVR(C=5, gamma="scale", epsilon=0.2)),
        "GaussProc": make_pipeline(StandardScaler(), GaussianProcessRegressor(
            kernel=ConstantKernel() * RBF() + WhiteKernel(), normalize_y=True, alpha=1e-3, random_state=0)),
        "RandomForest": RandomForestRegressor(500, min_samples_leaf=2, random_state=0, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(500, min_samples_leaf=2, random_state=0, n_jobs=-1),
        "GradBoost": GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.03, random_state=0),
        "HistGBM": HistGradientBoostingRegressor(max_iter=500, l2_regularization=1, random_state=0),
    }
    try:
        from xgboost import XGBRegressor
        m["XGBoost"] = XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.03,
                                    subsample=0.8, colsample_bytree=0.8, random_state=0)
    except Exception:
        pass
    try:
        from lightgbm import LGBMRegressor
        m["LightGBM"] = LGBMRegressor(n_estimators=500, num_leaves=31, learning_rate=0.03,
                                      subsample=0.8, random_state=0, verbose=-1)
    except Exception:
        pass
    return m


class MLP(nn.Module):
    def __init__(self, nin):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nin, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(Xtr, ytr, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    n = len(Xtr); idx = np.random.permutation(n); nv = max(20, int(0.15 * n))
    vi, ti = idx[:nv], idx[nv:]
    Xt, yt = torch.tensor(Xtr[ti], dtype=torch.float32), torch.tensor(ytr[ti], dtype=torch.float32)
    Xv, yv = torch.tensor(Xtr[vi], dtype=torch.float32), torch.tensor(ytr[vi], dtype=torch.float32)
    m = MLP(Xtr.shape[1]); opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=1e-3)
    lf = nn.SmoothL1Loss(); best, bs, bad = 1e9, None, 0
    for ep in range(500):
        m.train(); opt.zero_grad(); lf(m(Xt), yt).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = lf(m(Xv), yv).item()
        if vl < best - 1e-4:
            best, bs, bad = vl, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 40:
                break
    if bs:
        m.load_state_dict(bs)
    m.eval()
    return m


def mlp_predict(Xtr, ytr, Xte, n_ens=5):
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    ps = []
    for s in range(n_ens):
        m = train_mlp(Xtr_s, ytr, s)
        with torch.no_grad():
            ps.append(m(torch.tensor(Xte_s, dtype=torch.float32)).numpy())
    return np.mean(ps, axis=0)


def run(fit_pred, scheme):
    yt, yp = [], []
    if scheme == "kfold":
        for tri, tei in KFold(10, shuffle=True, random_state=0).split(d):
            yp.append(fit_pred(d.iloc[tri], d.iloc[tei])); yt.append(d.iloc[tei].ET_closed_mm.values)
    elif scheme == "year":
        for s in SITES:
            ds = d[d.SITE_ID == s]
            for y in sorted(ds.year.dropna().unique()):
                tr, te = ds[ds.year != y], ds[ds.year == y]
                if len(te) < 5 or len(tr) < 15:
                    continue
                yp.append(fit_pred(tr, te)); yt.append(te.ET_closed_mm.values)
    else:
        for s in SITES:
            tr, te = d[d.SITE_ID != s], d[d.SITE_ID == s]
            if len(te) < 5:
                continue
            yp.append(fit_pred(tr, te)); yt.append(te.ET_closed_mm.values)
    return r2_score(np.concatenate(yt), np.concatenate(yp))


def main():
    print(f"{len(d)} samples, {len(SITES)} sites, {len(FE)} features\n")
    rows = []
    for name, model in classical().items():
        def fp(tr, te, model=model):
            m = clone(model).fit(tr[FE].values, tr.ET_closed_mm.values)
            return m.predict(te[FE].values)
        rows.append((name, run(fp, "kfold"), run(fp, "year"), run(fp, "site")))
    # deep learning
    def fp_mlp(tr, te):
        return mlp_predict(tr[FE].values, tr.ET_closed_mm.values, te[FE].values)
    rows.append(("MLP-ensemble", run(fp_mlp, "kfold"), run(fp_mlp, "year"), run(fp_mlp, "site")))

    t = pd.DataFrame(rows, columns=["model", "kfold", "leave_year", "leave_site"]).sort_values(
        "leave_site", ascending=False)
    print(f"{'model':<14}{'K-fold':>9}{'leave-year':>12}{'leave-site':>12}")
    print("-" * 47)
    for _, r in t.iterrows():
        print(f"{r.model:<14}{r.kfold:>9.3f}{r.leave_year:>12.3f}{r.leave_site:>12.3f}")
    t.to_csv("/anvil/scratch/x-jwang120/coastal-et/data/processed/model_comparison_13.csv", index=False)
    print(f"\nbest leave-site (upscaling): {t.iloc[0].model} (R2={t.iloc[0].leave_site:.3f})")

    # ---- feature list + importance ----
    SAT = ["LAI", "EVI2", "SAVI", "NDVI", "NDWI", "MNDWI", "LST_K"]
    MET = ["TA_ERA", "VPD_ERA", "SW_IN_ERA", "WS_ERA", "ETo_mm", "DOY_sin", "DOY_cos"]
    print(f"\n=== {len(FE)} INPUT FEATURES ===")
    print("  Satellite (7):", SAT)
    print("  Meteorology (7):", MET)

    from sklearn.inspection import permutation_importance
    et = ExtraTreesRegressor(600, min_samples_leaf=2, random_state=0, n_jobs=-1).fit(
        d[FE].values, d.ET_closed_mm.values)
    gini = pd.Series(et.feature_importances_, index=FE)
    # permutation importance via leave-site-out (honest: on held-out sites)
    perm = np.zeros(len(FE))
    n = 0
    for s in SITES:
        tr, te = d[d.SITE_ID != s], d[d.SITE_ID == s]
        if len(te) < 5:
            continue
        m = ExtraTreesRegressor(400, min_samples_leaf=2, random_state=0, n_jobs=-1).fit(
            tr[FE].values, tr.ET_closed_mm.values)
        pi = permutation_importance(m, te[FE].values, te.ET_closed_mm.values,
                                    n_repeats=10, random_state=0)
        perm += np.clip(pi.importances_mean, 0, None)
        n += 1
    perm = pd.Series(perm / n, index=FE)
    imp = pd.DataFrame({"gini": gini, "permutation_LSO": perm}).sort_values("permutation_LSO")
    imp.to_csv("/anvil/scratch/x-jwang120/coastal-et/data/processed/feature_importance.csv")
    print("\n=== FEATURE IMPORTANCE (permutation, leave-site-out) ===")
    for f, r in imp.sort_values("permutation_LSO", ascending=False).iterrows():
        grp = "sat" if f in SAT else "met"
        print(f"  {f:<12} [{grp}]  perm={r.permutation_LSO:.3f}  gini={r.gini:.3f}")

    # ---- Nature-style feature-importance figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 3, "ytick.major.size": 3})
    INK = "#1a1a1a"
    col = ["#55A868" if f in SAT else "#4C72B0" for f in imp.index]
    fig, ax = plt.subplots(figsize=(4.6, 4.2), constrained_layout=True)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    y = np.arange(len(imp))
    ax.barh(y, imp.permutation_LSO.values, color=col, height=0.72, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(imp.index, fontsize=7.5)
    ax.set_xlabel("Permutation importance (leave-site-out $\\Delta R^2$)", fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK, labelsize=7)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc="#55A868", label="satellite"), Patch(fc="#4C72B0", label="meteorology")],
              frameon=False, fontsize=7, loc="lower right")
    ax.set_title("Feature importance for ET prediction (13 sites)", fontsize=9.5,
                 fontweight="bold", color=INK)
    for ext in ("png", "pdf"):
        fig.savefig(f"/anvil/scratch/x-jwang120/coastal-et/figures/feature_importance.{ext}",
                    dpi=450, facecolor="white", bbox_inches="tight")
    print("\nwrote feature_importance.png (+ .pdf)")


if __name__ == "__main__":
    main()
