"""Two diagnostics: (A) does real LST help? (B) is the model a DOY crutch?"""
import os, sys; sys.path.insert(0,"/anvil/scratch/x-jwang120/coastal-et/src")
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import train_et_models as T

data = T.build()
sites = ["US-Esm","US-TaS","US-Skr","US-Elm","US-EvM"]
def rf(): return RandomForestRegressor(n_estimators=600, min_samples_leaf=3, random_state=42, n_jobs=-1)

def loso(df, feats):
    df = df.dropna(subset=["ET_closed_mm"]+feats)
    preds=[]
    for s in sites:
        tr=df[df.SITE_ID!=s]; te=df[df.SITE_ID==s]
        if len(te)<8 or len(tr)<30: continue
        m=rf(); m.fit(tr[feats].values, tr.ET_closed_mm.values)
        preds.append(te.assign(p=m.predict(te[feats].values)))
    P=pd.concat(preds)
    return r2_score(P.ET_closed_mm,P.p), len(P)

ALL=[c for c in T.SAT+T.MET+T.EXTRA if c in data.columns]
NODOY=[c for c in ALL if not c.startswith("DOY")]
NOSAT=[c for c in ALL if c in T.MET+["ETo_mm","DOY_sin","DOY_cos"]]
SATONLY=[c for c in ALL if c in T.SAT]

print("=== (B) feature-group ablation, LOSO pooled R2 (window 500m) ===")
for name,f in [("ALL features",ALL),("no DOY (drop calendar)",NODOY),
               ("MET+ETo only (no satellite)",NOSAT),("satellite only",SATONLY)]:
    r,n=loso(data,f); print(f"  {name:<28} R2={r:>7.3f}  (n={n}, {len(f)} feats)")

print("\n=== (A) real-LST rows only vs interpolated ===")
strict = data[data.get('LST_age_d',pd.Series(99,index=data.index))==0]
print(f"  rows with LST_age_d==0 (real overpass): {len(strict)} / {len(data)}")
for name,sub in [("all rows (LST mostly interpolated)",data),
                 ("real-LST rows only",strict)]:
    r,n=loso(sub,ALL)
    print(f"  {name:<36} R2={r:>7.3f}  (n={n})")
# importance on real-LST rows
d2=strict.dropna(subset=["ET_closed_mm"]+ALL)
m=rf().fit(d2[ALL].values,d2.ET_closed_mm.values)
imp=pd.Series(m.feature_importances_,index=ALL).sort_values(ascending=False)
print("  real-LST importance: "+", ".join(f"{k}={v:.2f}" for k,v in imp.head(6).items()))
