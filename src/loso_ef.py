"""Fair MET-vs-satellite comparison + predict EVAPORATIVE FRACTION not raw ET."""
import os, sys; sys.path.insert(0,"/anvil/scratch/x-jwang120/coastal-et/src")
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import train_et_models as T
data=T.build()
sites=["US-Esm","US-TaS","US-Skr","US-Elm","US-EvM"]
def rf(): return RandomForestRegressor(n_estimators=600,min_samples_leaf=3,random_state=42,n_jobs=-1)
ALL=[c for c in T.SAT+T.MET+T.EXTRA if c in data.columns]
MET=[c for c in T.MET+["ETo_mm","DOY_sin","DOY_cos"] if c in data.columns]

def loso(df, feats, target, back_to_et=False):
    df=df.dropna(subset=[target,"ETo_mm"]+feats)
    P=[]
    for s in sites:
        tr=df[df.SITE_ID!=s]; te=df[df.SITE_ID==s]
        if len(te)<8 or len(tr)<30: continue
        m=rf(); m.fit(tr[feats].values, tr[target].values)
        pred=m.predict(te[feats].values)
        te=te.assign(p=pred)
        P.append(te)
    P=pd.concat(P)
    if back_to_et:   # score in ET space: ET = EF * ETo
        return r2_score(P.ET_closed_mm, P.p*P.ETo_mm), len(P)
    return r2_score(P[target], P.p), len(P)

# 1) FAIR comparison: same rows (those with satellite present)
sat_rows=data.dropna(subset=ALL)
print(f"=== FAIR comparison, identical rows (n={len(sat_rows.dropna(subset=['ET_closed_mm']))}) ===")
r_all,_=loso(sat_rows,ALL,"ET_closed_mm")
r_met,_=loso(sat_rows,MET,"ET_closed_mm")
print(f"  ALL features  R2={r_all:.3f}")
print(f"  MET+ETo only  R2={r_met:.3f}   (same rows -> honest satellite contribution)")

# 2) predict EVAPORATIVE FRACTION, score back in ET space
data["EF"]=data["ET_closed_mm"]/data["ETo_mm"].replace(0,np.nan)
data=data[(data.EF>0)&(data.EF<2.5)]
print(f"\n=== predict EF=ET/ETo, then ET=EF*ETo, LOSO (n rows vary) ===")
for name,f in [("ALL features",ALL),("MET only",[c for c in MET if c!='ETo_mm'])]:
    r,n=loso(data,f,"EF",back_to_et=True)
    print(f"  {name:<14} -> ET-space R2={r:>7.3f}  (n={n})")
# per-site for EF-ALL
print("\n  per-site (EF model, ALL features, scored in ET space):")
df=data.dropna(subset=["EF","ETo_mm"]+ALL)
for s in sites:
    tr=df[df.SITE_ID!=s]; te=df[df.SITE_ID==s]
    if len(te)<8: continue
    m=rf(); m.fit(tr[ALL].values,tr.EF.values)
    et_pred=m.predict(te[ALL].values)*te.ETo_mm.values
    print(f"    {s}: R2={r2_score(te.ET_closed_mm,et_pred):>7.3f}  MAE={mean_absolute_error(te.ET_closed_mm,et_pred):.2f}")
