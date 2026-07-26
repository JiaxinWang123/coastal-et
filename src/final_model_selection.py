"""Final model selection: after multicollinearity pruning, retrain the FULL model zoo
on the selected features, rank by spatial-transfer skill (leave-site-out), pick the
winner, refit it on all data, and save it as the production model.

Reports each model on both the full 14-feature set and the VIF-pruned set, so the choice
is transparent. Headline metric = pooled leave-site-out R2.
"""
import sys, warnings, json
sys.path.insert(0, "/anvil/scratch/x-jwang120/coastal-et/src")
import numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, HistGradientBoostingRegressor)
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_absolute_error

R = "/anvil/scratch/x-jwang120/coastal-et"
d = pd.read_parquet(f"{R}/data/processed/more_sites_table.parquet")
SAT = ["LAI", "EVI2", "SAVI", "NDVI", "NDWI", "MNDWI", "LST_K"]
MET = ["TA_ERA", "VPD_ERA", "SW_IN_ERA", "WS_ERA", "ETo_mm", "DOY_sin", "DOY_cos"]
FE = SAT + MET
SITES = sorted(d.SITE_ID.unique())


# ---------- multicollinearity feature selection (iterative VIF) ----------
def vif(cols):
    Z = StandardScaler().fit_transform(d[cols].dropna().values)
    return pd.Series(
        {f: 1.0 / max(1e-9, 1 - LinearRegression().fit(
            np.delete(Z, j, 1), Z[:, j]).score(np.delete(Z, j, 1), Z[:, j]))
         for j, f in enumerate(cols)}).sort_values(ascending=False)


def vif_select(cols, thresh=10.0):
    cols = list(cols)
    while len(cols) > 2 and vif(cols).max() > thresh:
        cols.remove(vif(cols).idxmax())
    return cols


SELECTED = vif_select(FE)
print(f"VIF-pruned feature set ({len(SELECTED)}): {SELECTED}\n")


# ---------- model zoo ----------
def zoo():
    z = {
        "Ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10)),
        "ElasticNet": lambda: make_pipeline(StandardScaler(), ElasticNet(alpha=0.05, l1_ratio=0.3, max_iter=5000)),
        "PLS": lambda: make_pipeline(StandardScaler(), PLSRegression(n_components=5)),
        "kNN": lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(10, weights="distance")),
        "SVR": lambda: make_pipeline(StandardScaler(), SVR(C=5, gamma="scale", epsilon=0.2)),
        "GaussProc": lambda: make_pipeline(StandardScaler(), GaussianProcessRegressor(
            kernel=ConstantKernel() * RBF() + WhiteKernel(), normalize_y=True, alpha=1e-3, random_state=0)),
        "RandomForest": lambda: RandomForestRegressor(600, max_features=0.5, min_samples_leaf=2, random_state=0, n_jobs=-1),
        "ExtraTrees": lambda: ExtraTreesRegressor(600, max_features=1.0, min_samples_leaf=2, random_state=0, n_jobs=-1),
        "GradBoost": lambda: GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.03, random_state=0),
        "HistGBM": lambda: HistGradientBoostingRegressor(max_iter=500, learning_rate=0.03, l2_regularization=1, random_state=0),
    }
    try:
        from xgboost import XGBRegressor
        z["XGBoost"] = lambda: XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.03,
                                            subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=0)
    except Exception:
        pass
    try:
        from lightgbm import LGBMRegressor
        z["LightGBM"] = lambda: LGBMRegressor(n_estimators=500, num_leaves=31, learning_rate=0.03,
                                              subsample=0.8, random_state=0, verbose=-1)
    except Exception:
        pass
    return z


def evaluate(mk, cols, scheme):
    yt, yp = [], []
    if scheme == "kfold":
        for tri, tei in KFold(10, shuffle=True, random_state=0).split(d):
            m = clone(mk()).fit(d.iloc[tri][cols].values, d.iloc[tri].ET_closed_mm.values)
            yp.append(m.predict(d.iloc[tei][cols].values)); yt.append(d.iloc[tei].ET_closed_mm.values)
    elif scheme == "year":
        for s in SITES:
            ds = d[d.SITE_ID == s]
            for y in sorted(ds.year.dropna().unique()):
                tr, te = ds[ds.year != y], ds[ds.year == y]
                if len(te) < 5 or len(tr) < 15:
                    continue
                m = clone(mk()).fit(tr[cols].values, tr.ET_closed_mm.values)
                yp.append(m.predict(te[cols].values)); yt.append(te.ET_closed_mm.values)
    else:
        for s in SITES:
            tr, te = d[d.SITE_ID != s], d[d.SITE_ID == s]
            if len(te) < 5:
                continue
            m = clone(mk()).fit(tr[cols].values, tr.ET_closed_mm.values)
            yp.append(m.predict(te[cols].values)); yt.append(te.ET_closed_mm.values)
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    return r2_score(yt, yp), mean_absolute_error(yt, yp)


# ---------- retrain the whole zoo on full vs pruned feature sets ----------
rows = []
for name, mk in zoo().items():
    for tag, cols in [("full14", FE), ("pruned", SELECTED)]:
        rk, _ = evaluate(mk, cols, "kfold")
        ry, _ = evaluate(mk, cols, "year")
        rs, mae = evaluate(mk, cols, "site")
        rows.append(dict(model=name, features=tag, n_feat=len(cols),
                         kfold=round(rk, 3), leave_year=round(ry, 3),
                         leave_site=round(rs, 3), leave_site_MAE=round(mae, 2)))
    print(f"  {name:<13} done", flush=True)

tab = pd.DataFrame(rows)
tab.to_csv(f"{R}/data/processed/final_model_selection.csv", index=False)
print("\n=== leave-site R2 (headline) — full14 vs pruned ===")
piv = tab.pivot(index="model", columns="features", values="leave_site").sort_values("pruned", ascending=False)
print(piv.to_string())

# ---------- pick the winner (best leave-site on the pruned set) and finalize ----------
best = tab[tab.features == "pruned"].sort_values("leave_site", ascending=False).iloc[0]
name = best.model
print(f"\nWINNER: {name}  (pruned features, leave-site R2={best.leave_site}, MAE={best.leave_site_MAE})")

prod = clone(zoo()[name]()).fit(d[SELECTED].values, d.ET_closed_mm.values)
meta = dict(model=name, features=SELECTED, n_train=len(d), n_sites=len(SITES),
            leave_site_R2=float(best.leave_site), leave_site_MAE=float(best.leave_site_MAE),
            kfold_R2=float(best.kfold), leave_year_R2=float(best.leave_year))
joblib.dump({**meta, "model": prod}, f"{R}/data/processed/final_model.joblib")  # fitted model wins the key
with open(f"{R}/data/processed/final_model.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"\nsaved production model -> data/processed/final_model.joblib")
print(json.dumps(meta, indent=2))
