"""Does adding LAI/EVI2/NDWI improve the daily ET gap-fill model (baseline R2~0.59)?

Same per-site leave-one-year-out gap-fill as gapfill_model.py, but the vegetation
features are footprint-weighted LAI/EVI2/NDWI/NDVI/MNDWI (from the per-pixel
record), interpolated to a daily axis (they move slowly), alongside met+ETo+season.

Compares three feature sets, so we can see exactly what the biophysical indices add:
  MET        met + ETo + season only
  +NDVI      add interpolated NDVI (the current production feature)
  +INDICES   add LAI + EVI2 + NDWI + MNDWI + LST too
"""
import os
import sys
sys.path.insert(0, "/anvil/scratch/x-jwang120/coastal-et/src")
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import gapfill_model as G

R = "/anvil/scratch/x-jwang120/coastal-et"
PIX = f"{R}/data/interim/pixels"
OUT = f"{R}/data/processed"
SITES = ["US-Esm", "US-TaS", "US-Skr", "US-Elm", "US-EvM"]
MET = ["TA_ERA", "VPD_ERA", "SW_IN_ERA", "WS_ERA", "ETo_mm", "DOY_sin", "DOY_cos"]
VEG = ["LAI", "EVI2", "NDWI", "NDVI", "MNDWI", "LST_K"]


def wavg(x, w):
    w = np.where(np.isfinite(x), w, 0.0)
    x = np.where(np.isfinite(x), x, 0.0)
    return x.dot(w) / w.sum() if w.sum() > 0 else np.nan


def build_site(site):
    et = pd.read_parquet(f"{OUT}/daily_closed_et.parquet")
    et.index = pd.to_datetime(et.index, utc=True)
    e = et[(et.SITE_ID == site) & (et.ET_closed_mm.notna())][["ET_closed_mm"]]
    e.index = e.index.normalize()
    e = e[(e.index >= "2018-01-01") & (e.index <= "2023-12-31")]
    if e.empty:
        return None

    px = pd.read_parquet(f"{PIX}/{site}_pixels.parquet")
    px["date"] = pd.to_datetime(px["date"], utc=True)
    savi = px["SAVI"].clip(0, 0.685)
    px["LAI"] = (-np.log((0.69 - savi) / 0.59) / 0.91).clip(0, 6)
    perov = px.groupby("date").apply(lambda x: pd.Series(
        {c: wavg(x[c].values, x.fp_weight.values) for c in VEG}))
    perov.index = perov.index.normalize()

    # interpolate each veg index onto a continuous daily axis (they vary slowly)
    full = pd.date_range("2018-01-01", "2023-12-31", freq="D", tz="UTC")
    veg = perov.reindex(full).interpolate(limit=30, limit_area="inside")

    d = e.join(veg, how="left")
    m = G.merged_met(site)
    if m is not None:
        d = d.join(m, how="left")
    gm = f"{R}/data/interim/gridmet/{site}_gridmet.parquet"
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
    return d


def rf():
    return RandomForestRegressor(400, min_samples_leaf=3, random_state=42, n_jobs=-1)


def loyo(data, feats):
    """leave-one-year-out within each site, pooled."""
    pr = []
    for s in SITES:
        ds = data[data.SITE_ID == s].dropna(subset=["ET_closed_mm"] + feats)
        for y in sorted(ds.year.unique()):
            tr = ds[ds.year != y]
            te = ds[ds.year == y]
            if len(te) < 10 or len(tr) < 40:
                continue
            m = rf()
            m.fit(tr[feats].values, tr.ET_closed_mm.values)
            pr.append(te.assign(pred=m.predict(te[feats].values)))
    P = pd.concat(pr)
    return r2_score(P.ET_closed_mm, P.pred), mean_absolute_error(P.ET_closed_mm, P.pred), len(P)


def main():
    data = pd.concat([build_site(s) for s in SITES if build_site(s) is not None])
    print(f"{len(data)} daily rows, {data.SITE_ID.nunique()} sites\n")
    print(f"{'feature set':<22}{'LOYO R2':>9}{'MAE':>7}{'n':>7}")
    print("-" * 46)
    for name, feats in [("MET only", MET),
                        ("MET + NDVI", MET + ["NDVI"]),
                        ("MET + all indices", MET + VEG)]:
        r, mae, n = loyo(data, feats)
        print(f"{name:<22}{r:>9.3f}{mae:>7.2f}{n:>7}")
    # FAIR: same rows (those with all indices present), MET-only vs MET+indices
    common = data.dropna(subset=["ET_closed_mm"] + MET + VEG)
    print("\n-- fair comparison, identical rows --")
    for name, feats in [("MET only (same rows)", MET),
                        ("MET + all indices", MET + VEG)]:
        r, mae, n = loyo(common, feats)
        print(f"{name:<22}{r:>9.3f}{mae:>7.2f}{n:>7}")

    # importance for the full set
    dd = data.dropna(subset=["ET_closed_mm"] + MET + VEG)
    m = rf()
    m.fit(dd[MET + VEG].values, dd.ET_closed_mm.values)
    imp = pd.Series(m.feature_importances_, index=MET + VEG).sort_values(ascending=False)
    print("\nRF importance (all features):",
          ", ".join(f"{k}={v:.2f}" for k, v in imp.head(8).items()))


if __name__ == "__main__":
    main()
