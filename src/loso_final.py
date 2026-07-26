import os, sys; sys.path.insert(0,"/anvil/scratch/x-jwang120/coastal-et/src")
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import train_et_models as T
data=T.build().dropna(subset=["ET_closed_mm"])
ALL=[c for c in T.SAT+T.MET+T.EXTRA if c in data.columns]
MET=[c for c in T.MET+["ETo_mm"] if c in data.columns]
def rf(): return RandomForestRegressor(n_estimators=600,min_samples_leaf=2,random_state=42,n_jobs=-1)
def agg(df,freq):
    if freq=="D": return df
    out=[]
    for s,g in df.groupby("SITE_ID"):
        r=g.resample(freq); m=r.mean(numeric_only=True); c=r["ET_closed_mm"].count()
        m=m[c>={"8D":3,"MS":12}[freq]]; m["SITE_ID"]=s; out.append(m)
    return pd.concat(out)
def loso(df,feats,sites):
    df=df.dropna(subset=["ET_closed_mm"]+feats); P=[]
    for s in sites:
        tr=df[df.SITE_ID!=s]; te=df[df.SITE_ID==s]
        if len(te)<6 or len(tr)<20: continue
        m=rf().fit(tr[feats].values,tr.ET_closed_mm.values)
        P.append(te.assign(p=m.predict(te[feats].values)))
    P=pd.concat(P); return r2_score(P.ET_closed_mm,P.p), mean_absolute_error(P.ET_closed_mm,P.p), len(P)
ALL5=["US-Esm","US-TaS","US-Skr","US-Elm","US-EvM"]
NOTAS=["US-Esm","US-Skr","US-Elm","US-EvM"]
print(f"{'timescale':<9}{'sites':<10}{'features':<10}{'R2':>8}{'MAE':>7}")
print("-"*44)
for freq,lab in [("D","daily"),("MS","monthly")]:
    a=agg(data,freq)
    for sset,sl in [(ALL5,"all 5"),(NOTAS,"drop TaS")]:
        for fs,fl in [(MET,"MET+ETo"),(ALL,"ALL")]:
            r,mae,n=loso(a,fs,sset)
            print(f"{lab:<9}{sl:<10}{fl:<10}{r:>8.3f}{mae:>7.2f}")
    print()
