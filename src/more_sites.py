"""Does adding coastal-wetland SITES rescue spatial upscaling (leave-tower-out)?

Uses tower-point (500 m window) satellite features for consistency across all
sites (footprints not available for the added sites; footprint weighting was
shown not to change the conclusion). Indices computed from the window-mean bands:
LAI (via SAVI), EVI2, SAVI, NDVI, NDWI, MNDWI, LST.

Compares leave-tower-out R2 for 5 Everglades sites vs 13 diverse coastal wetlands
(SC/LA/CA/NC/DE salt marsh, brackish, tidal forest, delta) -- the one lever left:
does ecosystem diversity let the satellite discriminate ET between sites?
"""
import os
import sys
sys.path.insert(0, "/anvil/scratch/x-jwang120/coastal-et/src")
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import train_indices_model as T   # for met_daily / merged met helpers via gapfill

R = "/anvil/scratch/x-jwang120/coastal-et"
TP = f"{R}/data/interim/tower_point"
EVERGLADES = ["US-Esm", "US-TaS", "US-Skr", "US-Elm", "US-EvM"]
ADDED = ["US-HB1", "US-HB4", "US-LA3", "US-EDN", "US-Myb", "US-Tw4", "US-NC4", "US-StJ"]
W = "500"
SATF = ["LAI", "EVI2", "SAVI", "NDVI", "NDWI", "MNDWI", "LST_K"]
MET = ["TA_ERA", "VPD_ERA", "SW_IN_ERA", "WS_ERA", "ETo_mm", "DOY_sin", "DOY_cos"]
FEATS = SATF + MET
import gapfill_model as G


def indices_from_bands(tp):
    """Compute the index set from tower-point 500 m window-mean bands."""
    def c(name):
        return tp[f"landsat_{name}_w{W}"] if f"landsat_{name}_w{W}" in tp else np.nan
    red, nir, grn, swr = c("red"), c("nir08"), c("green"), c("swir16")
    out = pd.DataFrame(index=tp.index)
    den = nir + red
    out["NDVI"] = ((nir - red) / den.where(den > 0.02)).clip(-1, 1)
    out["SAVI"] = (1.5 * (nir - red) / (nir + red + 0.5)).clip(-1, 1.5)
    out["EVI2"] = (2.5 * (nir - red) / (nir + 2.4 * red + 1)).clip(-1, 1.5)
    dw = nir + swr
    out["NDWI"] = ((nir - swr) / dw.where(dw > 0.02)).clip(-1, 1)
    dm = grn + swr
    out["MNDWI"] = ((grn - swr) / dm.where(dm > 0.02)).clip(-1, 1)
    savi = out["SAVI"].clip(0, 0.685)
    out["LAI"] = (-np.log((0.69 - savi) / 0.59) / 0.91).clip(0, 6)
    if f"LST_K_w{W}" in tp:
        out["LST_K"] = tp[f"LST_K_w{W}"]
    elif f"landsat_lwir11_w{W}" in tp:
        out["LST_K"] = tp[f"landsat_lwir11_w{W}"]
    return out


def build_site(site):
    p = f"{TP}/{site}_towerpoint.parquet"
    if not os.path.exists(p):
        return None
    tp = pd.read_parquet(p)
    tp.index = pd.to_datetime(tp.index, utc=True)
    tp = tp.groupby(tp.index.normalize()).mean()
    idx = indices_from_bands(tp)
    idx = idx[idx["LST_K"].notna()]              # real overpass days only
    if idx.empty:
        return None
    et = pd.read_parquet(f"{R}/data/processed/daily_closed_et.parquet")
    et.index = pd.to_datetime(et.index, utc=True)
    e = et[(et.SITE_ID == site) & (et.ET_closed_mm.notna())][["ET_closed_mm"]]
    e.index = e.index.normalize()
    m = G.merged_met(site)
    gm = f"{R}/data/interim/gridmet/{site}_gridmet.parquet"
    d = idx.join(e, how="inner")
    if m is not None:
        d = d.join(m, how="left")
    if os.path.exists(gm):
        g = pd.read_parquet(gm)
        g.index = pd.to_datetime(g.index, utc=True)
        if "ETo_mm" in g:
            d = d.join(g[["ETo_mm"]], how="left")
    doy = d.index.dayofyear
    d["DOY_sin"] = np.sin(2 * np.pi * doy / 365.25)
    d["DOY_cos"] = np.cos(2 * np.pi * doy / 365.25)
    d["SITE_ID"] = site
    d["year"] = d.index.year
    return d.dropna(subset=["ET_closed_mm"] + FEATS)


def gp():
    return make_pipeline(StandardScaler(), GaussianProcessRegressor(
        kernel=ConstantKernel() * RBF() + WhiteKernel(), normalize_y=True,
        alpha=1e-3, random_state=0))


def et_model():
    return ExtraTreesRegressor(400, min_samples_leaf=3, random_state=0, n_jobs=-1)


def evaluate(d, sites, mk):
    from sklearn.base import clone
    # random-CV
    yt, yp = [], []
    for tri, tei in KFold(10, shuffle=True, random_state=0).split(d):
        m = clone(mk()).fit(d.iloc[tri][FEATS].values, d.iloc[tri].ET_closed_mm.values)
        yp.append(m.predict(d.iloc[tei][FEATS].values)); yt.append(d.iloc[tei].ET_closed_mm.values)
    r_rand = r2_score(np.concatenate(yt), np.concatenate(yp))
    # leave-tower-out
    yt, yp = [], []
    for s in sites:
        tr, te = d[d.SITE_ID != s], d[d.SITE_ID == s]
        if len(te) < 5:
            continue
        m = clone(mk()).fit(tr[FEATS].values, tr.ET_closed_mm.values)
        yp.append(m.predict(te[FEATS].values)); yt.append(te.ET_closed_mm.values)
    r_tower = r2_score(np.concatenate(yt), np.concatenate(yp))
    return r_rand, r_tower


def main():
    all_sites = EVERGLADES + ADDED
    frames = []
    for s in all_sites:
        d = build_site(s)
        if d is not None and len(d) >= 10:
            frames.append(d)
            print(f"  {s}: {len(d)} overpass samples")
        else:
            print(f"  {s}: unavailable/too few")
    data = pd.concat(frames, ignore_index=True)
    have = sorted(data.SITE_ID.unique())
    print(f"\ntotal: {len(data)} samples, {len(have)} sites\n")

    ev = [s for s in EVERGLADES if s in have]
    d5 = data[data.SITE_ID.isin(ev)]
    print(f"{'set':<22}{'n':>6}{'sites':>7}{'random-CV':>11}{'leave-tower':>13}")
    print("-" * 60)
    for name, sub, sites in [("5 Everglades only", d5, ev),
                             (f"all {len(have)} coastal wetlands", data, have)]:
        for mn, mk in [("GP", gp), ("ExtraTrees", et_model)]:
            rr, rt = evaluate(sub, sites, mk)
            print(f"{name+' ['+mn+']':<22}{len(sub):>6}{len(sites):>7}{rr:>11.3f}{rt:>13.3f}")
    data.to_parquet(f"{R}/data/processed/more_sites_table.parquet")
    print(f"\nwrote more_sites_table.parquet")


if __name__ == "__main__":
    main()
