"""Does temporal aggregation rescue the satellite signal? daily vs 8-day vs monthly."""
import os, sys; sys.path.insert(0,"/anvil/scratch/x-jwang120/coastal-et/src")
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import train_et_models as T
data=T.build().dropna(subset=["ET_closed_mm"])
sites=["US-Esm","US-TaS","US-Skr","US-Elm","US-EvM"]
ALL=[c for c in T.SAT+T.MET+T.EXTRA if c in data.columns]
MET=[c for c in T.MET+["ETo_mm"] if c in data.columns]
def rf(): return RandomForestRegressor(n_estimators=600,min_samples_leaf=2,random_state=42,n_jobs=-1)

def agg(df, freq):
    if freq=="D": return df
    out=[]
    for s,g in df.groupby("SITE_ID"):
        # require >=40% of the period present, else the mean is unreliable
        r=g.resample(freq)
        m=r.mean(numeric_only=True)
        cnt=r["ET_closed_mm"].count()
        need={"8D":3,"MS":12}[freq]
        m=m[cnt>=need]; m["SITE_ID"]=s
        out.append(m)
    return pd.concat(out)

def loso(df,feats):
    df=df.dropna(subset=["ET_closed_mm"]+feats)
    P=[]
    for s in sites:
        tr=df[df.SITE_ID!=s]; te=df[df.SITE_ID==s]
        if len(te)<6 or len(tr)<20: continue
        m=rf().fit(tr[feats].values,tr.ET_closed_mm.values)
        P.append(te.assign(p=m.predict(te[feats].values)))
    P=pd.concat(P)
    return r2_score(P.ET_closed_mm,P.p), mean_absolute_error(P.ET_closed_mm,P.p), len(P)

print(f"{'timescale':<10}{'features':<12}{'LOSO R2':>9}{'MAE':>7}{'n':>6}")
print("-"*46)
for freq,lab in [("D","daily"),("8D","8-day"),("MS","monthly")]:
    a=agg(data,freq)
    for fs,fl in [(ALL,"ALL"),(MET,"MET+ETo")]:
        r,mae,n=loso(a,fs)
        print(f"{lab:<10}{fl:<12}{r:>9.3f}{mae:>7.2f}{n:>6}")
    print()
# per-site at monthly, ALL
print("=== monthly, ALL features, per site ===")
a=agg(data,"MS").dropna(subset=["ET_closed_mm"]+ALL)
for s in sites:
    tr=a[a.SITE_ID!=s]; te=a[a.SITE_ID==s]
    if len(te)<6: continue
    m=rf().fit(tr[ALL].values,tr.ET_closed_mm.values)
    p=m.predict(te[ALL].values)
    print(f"  {s}: R2={r2_score(te.ET_closed_mm,p):>7.3f}  MAE={mean_absolute_error(te.ET_closed_mm,p):.2f}  n={len(te)}")
