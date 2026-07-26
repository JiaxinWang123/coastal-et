"""Is the MODEL the bottleneck, or the signal? Sweep many algorithms.
Config: monthly, 500 m window, leave-one-site-out, drop-TaS pool."""
import os, sys; sys.path.insert(0,"/anvil/scratch/x-jwang120/coastal-et/src")
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
    HistGradientBoostingRegressor, GradientBoostingRegressor)
from sklearn.linear_model import Ridge, LinearRegression, HuberRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import train_et_models as T

data=T.build().dropna(subset=["ET_closed_mm"])
ALL=[c for c in T.SAT+T.MET+T.EXTRA if c in data.columns]
MET=[c for c in T.MET+["ETo_mm"] if c in data.columns]
def agg(df):
    out=[]
    for s,g in df.groupby("SITE_ID"):
        r=g.resample("MS"); m=r.mean(numeric_only=True); c=r["ET_closed_mm"].count()
        m=m[c>=12]; m["SITE_ID"]=s; out.append(m)
    return pd.concat(out)
mon=agg(data)
SITES=["US-Esm","US-Skr","US-Elm","US-EvM"]   # drop-TaS pool

def sc(): return StandardScaler()
MODELS={
 "Linear":            lambda: make_pipeline(sc(),LinearRegression()),
 "Ridge(a=10)":       lambda: make_pipeline(sc(),Ridge(alpha=10)),
 "Huber":             lambda: make_pipeline(sc(),HuberRegressor(max_iter=2000)),
 "kNN(k=10)":         lambda: make_pipeline(sc(),KNeighborsRegressor(10)),
 "SVR(rbf)":          lambda: make_pipeline(sc(),SVR(C=3,gamma="scale")),
 "RandomForest":      lambda: RandomForestRegressor(600,min_samples_leaf=2,random_state=42,n_jobs=-1),
 "ExtraTrees":        lambda: ExtraTreesRegressor(600,min_samples_leaf=2,random_state=42,n_jobs=-1),
 "HistGBM":           lambda: HistGradientBoostingRegressor(max_iter=400,l2_regularization=1,random_state=42),
 "GradBoost":         lambda: GradientBoostingRegressor(n_estimators=300,max_depth=2,random_state=42),
 "MLP(64,32)":        lambda: make_pipeline(sc(),MLPRegressor((64,32),alpha=1e-2,max_iter=3000,early_stopping=True,random_state=42)),
}
def loso(df,feats,mk):
    df=df.dropna(subset=["ET_closed_mm"]+feats); P=[]
    for s in SITES:
        tr=df[df.SITE_ID!=s]; te=df[df.SITE_ID==s]
        if len(te)<6 or len(tr)<20: continue
        m=mk(); m.fit(tr[feats].values,tr.ET_closed_mm.values)
        P.append(te.assign(p=m.predict(te[feats].values)))
    P=pd.concat(P); return r2_score(P.ET_closed_mm,P.p), mean_absolute_error(P.ET_closed_mm,P.p)

print("monthly, drop-TaS, leave-one-site-out\n")
print(f"{'model':<16}{'MET+ETo R2':>12}{'MAE':>7}   {'ALL R2':>9}{'MAE':>7}")
print("-"*54)
for name,mk in MODELS.items():
    rm,mm=loso(mon,MET,mk); ra,ma=loso(mon,ALL,mk)
    print(f"{name:<16}{rm:>12.3f}{mm:>7.2f}   {ra:>9.3f}{ma:>7.2f}")
