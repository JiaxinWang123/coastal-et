"""Broad model comparison for the predictive ET task.

Same footprint-weighted overpass features + 3 evaluations as predict_et.py:
  random-CV (monitored sites) | leave-year-out (temporal) | leave-tower-out (spatial)
Adds gradient-boosting libraries, Gaussian process, PLS, ElasticNet, tuned SVR/MLP.
"""
import os
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
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import train_indices_model as T

SITES = ["US-Esm", "US-TaS", "US-Skr", "US-Elm", "US-EvM"]
FEATS = T.FEATS
SEED = 0


def sc_pipe():
    return StandardScaler()


def build_models():
    m = {
        "Ridge": make_pipeline(sc_pipe(), Ridge(alpha=10)),
        "ElasticNet": make_pipeline(sc_pipe(), ElasticNet(alpha=0.05, l1_ratio=0.3, max_iter=5000)),
        "PLS(5)": make_pipeline(sc_pipe(), PLSRegression(n_components=5)),
        "kNN(10)": make_pipeline(sc_pipe(), KNeighborsRegressor(10, weights="distance")),
        "SVR(rbf)": make_pipeline(sc_pipe(), SVR(C=5, gamma="scale", epsilon=0.2)),
        "MLP(64,32)": make_pipeline(sc_pipe(), MLPRegressor((64, 32), alpha=1e-2,
                        max_iter=4000, early_stopping=True, random_state=SEED)),
        "GaussProc": make_pipeline(sc_pipe(), GaussianProcessRegressor(
                        kernel=ConstantKernel() * RBF() + WhiteKernel(),
                        normalize_y=True, alpha=1e-3, random_state=SEED)),
        "RandomForest": RandomForestRegressor(400, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(400, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
        "GradBoost": GradientBoostingRegressor(n_estimators=300, max_depth=2, random_state=SEED),
        "HistGBM": HistGradientBoostingRegressor(max_iter=400, l2_regularization=1, random_state=SEED),
    }
    try:
        from xgboost import XGBRegressor
        m["XGBoost"] = XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.03,
                                    subsample=0.8, colsample_bytree=0.8, random_state=SEED)
    except Exception as e:
        print("  (xgboost unavailable:", e, ")")
    try:
        from lightgbm import LGBMRegressor
        m["LightGBM"] = LGBMRegressor(n_estimators=400, num_leaves=15, learning_rate=0.03,
                                      subsample=0.8, random_state=SEED, verbose=-1)
    except Exception as e:
        print("  (lightgbm unavailable:", e, ")")
    return m


def clone_fit_pred(model, Xtr, ytr, Xte):
    from sklearn.base import clone
    m = clone(model)
    m.fit(Xtr, ytr)
    p = m.predict(Xte)
    return np.asarray(p).ravel()


def r_random(d, model, k=10):
    kf = KFold(n_splits=k, shuffle=True, random_state=SEED)
    yt, yp = [], []
    for tri, tei in kf.split(d):
        tr, te = d.iloc[tri], d.iloc[tei]
        yp.append(clone_fit_pred(model, tr[FEATS].values, tr.ET_closed_mm.values, te[FEATS].values))
        yt.append(te.ET_closed_mm.values)
    return r2_score(np.concatenate(yt), np.concatenate(yp))


def r_year(d, model):
    yt, yp = [], []
    for s in SITES:
        ds = d[d.SITE_ID == s]
        for y in sorted(ds.year.unique()):
            tr, te = ds[ds.year != y], ds[ds.year == y]
            if len(te) < 5 or len(tr) < 20:
                continue
            yp.append(clone_fit_pred(model, tr[FEATS].values, tr.ET_closed_mm.values, te[FEATS].values))
            yt.append(te.ET_closed_mm.values)
    return r2_score(np.concatenate(yt), np.concatenate(yp))


def r_tower(d, model):
    yt, yp = [], []
    for s in SITES:
        tr, te = d[d.SITE_ID != s], d[d.SITE_ID == s]
        if len(te) < 5:
            continue
        yp.append(clone_fit_pred(model, tr[FEATS].values, tr.ET_closed_mm.values, te[FEATS].values))
        yt.append(te.ET_closed_mm.values)
    return r2_score(np.concatenate(yt), np.concatenate(yp))


def main():
    d = T.build().dropna(subset=["ET_closed_mm"] + FEATS).reset_index(drop=True)
    print(f"{len(d)} samples, {len(FEATS)} features, {d.SITE_ID.nunique()} sites\n")
    rows = []
    for name, model in build_models().items():
        try:
            rows.append((name, r_random(d, model), r_year(d, model), r_tower(d, model)))
        except Exception as e:
            print(f"  {name}: FAILED {str(e)[:60]}")
    tab = pd.DataFrame(rows, columns=["model", "random_CV", "leave_year", "leave_tower"])
    tab = tab.sort_values("leave_year", ascending=False)
    print(f"{'model':<14}{'random-CV':>11}{'leave-year':>12}{'leave-tower':>13}")
    print("-" * 50)
    for _, r in tab.iterrows():
        print(f"{r.model:<14}{r.random_CV:>11.3f}{r.leave_year:>12.3f}{r.leave_tower:>13.3f}")
    tab.to_csv(f"/anvil/scratch/x-jwang120/coastal-et/data/processed/model_comparison.csv", index=False)
    print(f"\nbest leave-year:  {tab.iloc[0].model} ({tab.iloc[0].leave_year:.3f})")
    print(f"best random-CV:   {tab.sort_values('random_CV').iloc[-1].model} "
          f"({tab.random_CV.max():.3f})")


if __name__ == "__main__":
    main()
