"""Leave-one-site-out CV across all 5 Everglades sites, all years.

The fixed 3-train/2-test split put the two HIGHEST-ET sites (Elm 4.14, EvM 4.33)
in test, above the training pool max -- and a random forest cannot extrapolate
past its training target range, so that split scores negative R2 by construction.

LOSO rotates every site through the held-out position, so each site is predicted
by a model that HAS seen the full ET range. It is the fair "transfer to an unseen
site" test, and it also exposes WHICH ecosystems are hard.

Reports the pooled R2 and per-site R2 for RF and MLP, plus the met-only and ETo
baselines each fold must beat.
"""
import os
os.environ.setdefault("COASTAL_ET_WIN", os.environ.get("COASTAL_ET_WIN", "500"))
import sys; sys.path.insert(0, "/anvil/scratch/x-jwang120/coastal-et/src")
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import train_et_models as T

data = T.build()
feats = [c for c in T.SAT + T.MET + T.EXTRA if c in data.columns]
need = ["ET_closed_mm"] + feats
data = data.dropna(subset=need)
sites = ["US-Esm", "US-TaS", "US-Skr", "US-Elm", "US-EvM"]

def rf():  return RandomForestRegressor(n_estimators=600, min_samples_leaf=3,
                                        random_state=T.SEED, n_jobs=-1)
def mlp(): return make_pipeline(StandardScaler(),
             MLPRegressor(hidden_layer_sizes=(64,32), alpha=1e-2, max_iter=3000,
                          early_stopping=True, n_iter_no_change=30, random_state=T.SEED))

met_f = [c for c in T.MET + ["DOY_sin","DOY_cos"] if c in feats]
rows, pred_all = [], []
print(f"WINDOW={os.environ['COASTAL_ET_WIN']} m   features={len(feats)}   n={len(data)}\n")
print(f"{'held-out site':<10}{'n':>5}{'RF R2':>8}{'MLP R2':>8}{'met-only':>10}{'ETo':>8}{'RF bias':>9}")
print("-"*60)
for s in sites:
    tr = data[data.SITE_ID != s]; te = data[data.SITE_ID == s]
    if len(te) < 10: continue
    Xtr, ytr = tr[feats].values, tr.ET_closed_mm.values
    Xte, yte = te[feats].values, te.ET_closed_mm.values
    m = rf(); m.fit(Xtr, ytr); p_rf = m.predict(Xte)
    ml = mlp(); ml.fit(Xtr, ytr); p_ml = ml.predict(Xte)
    mo = rf(); mo.fit(tr[met_f].values, ytr); p_mo = mo.predict(te[met_f].values)
    r2 = lambda p: r2_score(yte, p)
    rows.append(dict(site=s, n=len(te), rf=r2(p_rf), mlp=r2(p_ml),
                     met=r2(p_mo), eto=r2(te.ETo_mm.values) if "ETo_mm" in te else np.nan,
                     bias=float(np.mean(p_rf-yte))))
    print(f"{s:<10}{len(te):>5}{r2(p_rf):>8.3f}{r2(p_ml):>8.3f}{r2(p_mo):>10.3f}"
          f"{(r2(te.ETo_mm.values) if 'ETo_mm' in te else np.nan):>8.3f}{np.mean(p_rf-yte):>9.2f}")
    pred_all.append(te.assign(pred_rf=p_rf, pred_mlp=p_ml))

# POOLED: concatenate all held-out predictions, score once
P = pd.concat(pred_all)
print("-"*60)
print(f"{'POOLED':<10}{len(P):>5}{r2_score(P.ET_closed_mm,P.pred_rf):>8.3f}"
      f"{r2_score(P.ET_closed_mm,P.pred_mlp):>8.3f}")
print(f"\n  pooled RF : R2={r2_score(P.ET_closed_mm,P.pred_rf):.3f}  "
      f"RMSE={np.sqrt(mean_squared_error(P.ET_closed_mm,P.pred_rf)):.2f}  "
      f"MAE={mean_absolute_error(P.ET_closed_mm,P.pred_rf):.2f}")

r = pd.DataFrame(rows)
r.to_csv(f"{T.OUT}/loso_results.csv", index=False)
P.to_parquet(f"{T.OUT}/loso_predictions.parquet")
print(f"\n  wrote loso_results.csv, loso_predictions.parquet")
imp = pd.Series(rf().fit(data[feats].values, data.ET_closed_mm.values).feature_importances_,
                index=feats).sort_values(ascending=False)
print("\n  RF importance (all data): " + ", ".join(f"{k}={v:.2f}" for k,v in imp.head(6).items()))
