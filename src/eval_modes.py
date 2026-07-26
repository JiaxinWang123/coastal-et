"""What CAN we predict with the current data? Three honest evaluation modes.

  A. LOSO (spatial transfer)      predict an UNSEEN site        -- the hard goal
  B. Leave-one-year-out (temporal) predict UNSEEN YEARS at KNOWN sites -- gap-fill
  C. Within-site random CV         explain ET variation at a site  -- upper bound
     (blocked by month to avoid day-to-day autocorrelation leakage)

Same features, same models. The gap between A and B/C is the whole story: it
tells us this data supports GAP-FILLING known towers, not SCALING to new ones.
"""
import os, sys; sys.path.insert(0,"/anvil/scratch/x-jwang120/coastal-et/src")
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
import train_et_models as T

data=T.build().dropna(subset=["ET_closed_mm"])
SITES=["US-Esm","US-TaS","US-Skr","US-Elm","US-EvM"]
ALL=[c for c in T.SAT+T.MET+T.EXTRA if c in data.columns]
def rf(): return RandomForestRegressor(400,min_samples_leaf=2,random_state=42,n_jobs=-1)
def rg(): return make_pipeline(StandardScaler(),Ridge(alpha=10))
data=data.dropna(subset=ALL).copy()
data["year"]=data.index.year
data["ym"]=data.index.to_period("M").astype(str)

def fit_pred(tr,te,mk):
    m=mk(); m.fit(tr[ALL].values,tr.ET_closed_mm.values); return m.predict(te[ALL].values)

def modeA(mk):  # LOSO spatial
    P=[]
    for s in SITES:
        tr=data[data.SITE_ID!=s]; te=data[data.SITE_ID==s]
        if len(te)<10: continue
        P.append(te.assign(p=fit_pred(tr,te,mk)))
    P=pd.concat(P); return r2_score(P.ET_closed_mm,P.p),mean_absolute_error(P.ET_closed_mm,P.p)

def modeB(mk):  # leave-one-year-out, within each site (temporal gap-fill)
    P=[]
    for s in SITES:
        d=data[data.SITE_ID==s]
        for y in sorted(d.year.unique()):
            tr=d[d.year!=y]; te=d[d.year==y]
            if len(te)<10 or len(tr)<30: continue
            P.append(te.assign(p=fit_pred(tr,te,mk)))
    P=pd.concat(P); return r2_score(P.ET_closed_mm,P.p),mean_absolute_error(P.ET_closed_mm,P.p)

def modeC(mk):  # within-site, month-blocked K-fold (upper bound, low leakage)
    P=[]
    for s in SITES:
        d=data[data.SITE_ID==s]
        if d.ym.nunique()<5: continue
        gkf=GroupKFold(n_splits=min(5,d.ym.nunique()))
        for tri,tei in gkf.split(d,groups=d.ym):
            tr=d.iloc[tri]; te=d.iloc[tei]
            P.append(te.assign(p=fit_pred(tr,te,mk)))
    P=pd.concat(P); return r2_score(P.ET_closed_mm,P.p),mean_absolute_error(P.ET_closed_mm,P.p)

print(f"daily rows: {len(data)}, sites: {data.SITE_ID.nunique()}, features: {len(ALL)}\n")
print(f"{'evaluation mode':<40}{'RF R2':>8}{'RF MAE':>8}{'Ridge R2':>10}{'Ridge MAE':>10}")
print("-"*76)
for lab,fn in [("A. predict UNSEEN SITE (spatial transfer)",modeA),
               ("B. predict UNSEEN YEAR at known site (gap-fill)",modeB),
               ("C. within-site, month-blocked CV (upper bound)",modeC)]:
    rr,rm=fn(rf); gr,gm=fn(rg)
    print(f"{lab:<40}{rr:>8.3f}{rm:>8.2f}{gr:>10.3f}{gm:>10.2f}")
print()
# per-site gap-fill skill (mode B), the operationally useful number
print("=== gap-fill skill per site (leave-one-year-out, RF) ===")
for s in SITES:
    d=data[data.SITE_ID==s]; P=[]
    for y in sorted(d.year.unique()):
        tr=d[d.year!=y]; te=d[d.year==y]
        if len(te)<10 or len(tr)<30: continue
        P.append(te.assign(p=fit_pred(tr,te,rf)))
    if not P: continue
    P=pd.concat(P)
    print(f"  {s}: R2={r2_score(P.ET_closed_mm,P.p):>6.3f}  MAE={mean_absolute_error(P.ET_closed_mm,P.p):.2f} mm  (n={len(P)})")
