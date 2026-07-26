"""Ablation: how much does each feature group contribute to daily-ET skill,
and is ETo doing all the work? ExtraTrees on the 13-site table, 3 CV schemes.
"""
import sys
sys.path.insert(0, "/anvil/scratch/x-jwang120/coastal-et/src")
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import r2_score

d = pd.read_parquet("/anvil/scratch/x-jwang120/coastal-et/data/processed/more_sites_table.parquet")
SITES = sorted(d.SITE_ID.unique())
MET = ["TA_ERA", "VPD_ERA", "SW_IN_ERA", "WS_ERA", "ETo_mm", "DOY_sin", "DOY_cos"]
MET_noETo = [f for f in MET if f != "ETo_mm"]
SAT = ["LAI", "EVI2", "SAVI", "NDVI", "NDWI", "MNDWI", "LST_K"]
SAT_water = ["NDWI", "MNDWI", "LST_K"]              # water/thermal only
SAT_green = ["LAI", "EVI2", "SAVI", "NDVI"]          # greenness only

CONFIGS = {
    "ETo only (1)": ["ETo_mm"],
    "MET no ETo (6)": MET_noETo,
    "MET all (7)": MET,
    "SAT green (4)": SAT_green,
    "SAT water+LST (3)": SAT_water,
    "SAT all (7)": SAT,
    "SAT + ETo (8)": SAT + ["ETo_mm"],
    "FULL (14)": SAT + MET,
}


def mk():
    return ExtraTreesRegressor(400, min_samples_leaf=2, random_state=0, n_jobs=-1)


def evalR2(feats, scheme):
    yt, yp = [], []
    if scheme == "kfold":
        for tri, tei in KFold(10, shuffle=True, random_state=0).split(d):
            m = clone(mk()).fit(d.iloc[tri][feats].values, d.iloc[tri].ET_closed_mm.values)
            yp.append(m.predict(d.iloc[tei][feats].values)); yt.append(d.iloc[tei].ET_closed_mm.values)
    elif scheme == "year":
        for s in SITES:
            ds = d[d.SITE_ID == s]
            for y in sorted(ds.year.dropna().unique()):
                tr, te = ds[ds.year != y], ds[ds.year == y]
                if len(te) < 5 or len(tr) < 15:
                    continue
                m = clone(mk()).fit(tr[feats].values, tr.ET_closed_mm.values)
                yp.append(m.predict(te[feats].values)); yt.append(te.ET_closed_mm.values)
    else:
        for s in SITES:
            tr, te = d[d.SITE_ID != s], d[d.SITE_ID == s]
            if len(te) < 5:
                continue
            m = clone(mk()).fit(tr[feats].values, tr.ET_closed_mm.values)
            yp.append(m.predict(te[feats].values)); yt.append(te.ET_closed_mm.values)
    return r2_score(np.concatenate(yt), np.concatenate(yp))


print(f"{len(d)} samples, {len(SITES)} sites\n")
print(f"{'feature group':<20}{'K-fold':>9}{'leave-year':>12}{'leave-site':>12}")
print("-" * 53)
rows = []
for name, feats in CONFIGS.items():
    rk, ry, rs = evalR2(feats, "kfold"), evalR2(feats, "year"), evalR2(feats, "site")
    rows.append((name, rk, ry, rs))
    print(f"{name:<20}{rk:>9.3f}{ry:>12.3f}{rs:>12.3f}", flush=True)
pd.DataFrame(rows, columns=["group", "kfold", "leave_year", "leave_site"]).to_csv(
    "/anvil/scratch/x-jwang120/coastal-et/data/processed/feature_group_ablation.csv", index=False)
print("\nwrote feature_group_ablation.csv")
