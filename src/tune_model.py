"""Hyperparameter search — tuned for SPATIAL TRANSFER (pooled leave-site-out R2),
not K-fold. K-fold-tuned models overfit interpolation and don't transfer to new towers.

Reports: default ExtraTrees vs the best tuned ExtraTrees, plus tuned RF / HistGBM,
and — as a reframing test — predicting Kc = ET/ETo instead of ET directly.
"""
import sys, itertools, warnings
sys.path.insert(0, "/anvil/scratch/x-jwang120/coastal-et/src")
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import r2_score

d = pd.read_parquet("/anvil/scratch/x-jwang120/coastal-et/data/processed/more_sites_table.parquet")
FE = ["LAI","EVI2","SAVI","NDVI","NDWI","MNDWI","LST_K","TA_ERA","VPD_ERA","SW_IN_ERA","WS_ERA","ETo_mm","DOY_sin","DOY_cos"]
SITES = sorted(d.SITE_ID.unique())


def leave_site(mk, target="ET", eto_floor=0.3):
    yt, yp = [], []
    for s in SITES:
        tr, te = d[d.SITE_ID != s], d[d.SITE_ID == s]
        if len(te) < 5:
            continue
        if target == "ET":
            m = clone(mk()).fit(tr[FE].values, tr.ET_closed_mm.values)
            p = m.predict(te[FE].values)
        else:  # predict Kc = ET/ETo, then multiply back
            kc = (tr.ET_closed_mm / tr.ETo_mm.clip(lower=eto_floor)).values
            m = clone(mk()).fit(tr[FE].values, kc)
            p = m.predict(te[FE].values) * te.ETo_mm.clip(lower=eto_floor).values
        yp.append(p); yt.append(te.ET_closed_mm.values)
    return r2_score(np.concatenate(yt), np.concatenate(yp))


def kfold(mk):
    yt, yp = [], []
    for tri, tei in KFold(10, shuffle=True, random_state=0).split(d):
        m = clone(mk()).fit(d.iloc[tri][FE].values, d.iloc[tri].ET_closed_mm.values)
        yp.append(m.predict(d.iloc[tei][FE].values)); yt.append(d.iloc[tei].ET_closed_mm.values)
    return r2_score(np.concatenate(yt), np.concatenate(yp))


default = lambda: ExtraTreesRegressor(400, min_samples_leaf=2, random_state=0, n_jobs=-1)
print(f"DEFAULT ExtraTrees   leave-site={leave_site(default):.3f}  kfold={kfold(default):.3f}\n")

print("=== ExtraTrees grid (selection metric = pooled leave-site R2) ===")
best = (None, -9)
for ne, mf, msl in itertools.product([400, 800, 1200], ["sqrt", 0.33, 0.5, 1.0], [1, 2, 4]):
    mk = (lambda ne=ne, mf=mf, msl=msl:
          ExtraTreesRegressor(ne, max_features=mf, min_samples_leaf=msl, random_state=0, n_jobs=-1))
    rs = leave_site(mk)
    if rs > best[1]:
        best = ((ne, mf, msl), rs)
    print(f"  n={ne:<5} max_features={str(mf):<5} min_leaf={msl}  leave-site={rs:.3f}", flush=True)
print(f"\nBEST ExtraTrees: n={best[0][0]}, max_features={best[0][1]}, min_leaf={best[0][2]}"
      f"  leave-site={best[1]:.3f}")
bne, bmf, bml = best[0]
bestmk = lambda: ExtraTreesRegressor(bne, max_features=bmf, min_samples_leaf=bml, random_state=0, n_jobs=-1)
print(f"  (its kfold={kfold(bestmk):.3f})\n")

print("=== other tuned families (leave-site) ===")
rf = lambda: RandomForestRegressor(800, max_features=0.33, min_samples_leaf=2, random_state=0, n_jobs=-1)
hg = lambda: HistGradientBoostingRegressor(max_iter=600, learning_rate=0.03, max_leaf_nodes=15,
                                           l2_regularization=2, random_state=0)
print(f"  RandomForest(tuned)  leave-site={leave_site(rf):.3f}")
print(f"  HistGBM(tuned)       leave-site={leave_site(hg):.3f}")

print("\n=== reframing: predict Kc=ET/ETo then x ETo (best ExtraTrees) ===")
print(f"  target=ET   leave-site={leave_site(bestmk, 'ET'):.3f}")
print(f"  target=Kc   leave-site={leave_site(bestmk, 'Kc'):.3f}")
