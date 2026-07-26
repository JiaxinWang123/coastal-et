"""Build the footprint-weighted 13-site modelling table and re-validate.

For every tower + overpass, we take the per-pixel indices (extract_pixels.py output)
and collapse them to a single FOOTPRINT-WEIGHTED value per index (weights = Kljun-2015
footprint contribution), instead of a 500 m window mean. LAI is derived from the
footprint-weighted SAVI. Meteorology / ET / ETo are joined exactly as in more_sites.

Writes data/processed/more_sites_table_fp.parquet and prints leave-site-out R2 for the
production model (ExtraTrees on the 11 VIF-pruned features), footprint vs 500 m window.
"""
import os, sys
sys.path.insert(0, "/anvil/scratch/x-jwang120/coastal-et/src")
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import more_sites as M
import gapfill_model as G
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error

R = "/anvil/scratch/x-jwang120/coastal-et"
PIX = f"{R}/data/interim/pixels"
IDX = ["NDVI", "SAVI", "EVI2", "NDWI", "MNDWI", "LST_K"]
SITES = M.EVERGLADES + M.ADDED
PRUNED = ["LAI", "NDVI", "MNDWI", "LST_K", "TA_ERA", "VPD_ERA", "SW_IN_ERA", "WS_ERA", "ETo_mm", "DOY_sin", "DOY_cos"]


def fp_weighted(site):
    p = f"{PIX}/{site}_pixels.parquet"
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df.date, utc=True).dt.normalize()
    rows = []
    for date, g in df.groupby("date"):
        r = {"date": date}
        for c in IDX:
            sub = g[["fp_weight", c]].replace([np.inf, -np.inf], np.nan).dropna()
            sub = sub[sub.fp_weight > 0]
            if len(sub) and sub.fp_weight.sum() > 0:
                r[c] = float(np.average(sub[c].values, weights=sub.fp_weight.values))
        rows.append(r)
    out = pd.DataFrame(rows).set_index("date")
    savi = out["SAVI"].clip(0, 0.685)
    out["LAI"] = (-np.log((0.69 - savi) / 0.59) / 0.91).clip(0, 6)
    return out[out["LST_K"].notna()]


def build_site_fp(site):
    idx = fp_weighted(site)
    if idx is None or idx.empty:
        return None
    et = pd.read_parquet(f"{R}/data/processed/daily_closed_et.parquet")
    et.index = pd.to_datetime(et.index, utc=True)
    e = et[(et.SITE_ID == site) & (et.ET_closed_mm.notna())][["ET_closed_mm"]]
    e.index = e.index.normalize()
    m = G.merged_met(site)
    d = idx.join(e, how="inner")
    if m is not None:
        d = d.join(m, how="left")
    gm = f"{R}/data/interim/gridmet/{site}_gridmet.parquet"
    if os.path.exists(gm):
        g = pd.read_parquet(gm); g.index = pd.to_datetime(g.index, utc=True)
        if "ETo_mm" in g:
            d = d.join(g[["ETo_mm"]], how="left")
    doy = d.index.dayofyear
    d["DOY_sin"] = np.sin(2 * np.pi * doy / 365.25)
    d["DOY_cos"] = np.cos(2 * np.pi * doy / 365.25)
    d["SITE_ID"] = site; d["year"] = d.index.year
    return d.dropna(subset=["ET_closed_mm"] + M.FEATS)


def leave_site(d, feats):
    yt, yp = [], []
    for s in sorted(d.SITE_ID.unique()):
        tr, te = d[d.SITE_ID != s], d[d.SITE_ID == s]
        if len(te) < 5:
            continue
        m = ExtraTreesRegressor(400, min_samples_leaf=2, random_state=0, n_jobs=-1)
        m.fit(tr[feats].values, tr.ET_closed_mm.values)
        yp.append(m.predict(te[feats].values)); yt.append(te.ET_closed_mm.values)
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    return r2_score(yt, yp), mean_absolute_error(yt, yp)


def main():
    frames = []
    for s in SITES:
        d = build_site_fp(s)
        n = 0 if d is None else len(d)
        print(f"  {s:<8} {n:>4} footprint overpass rows", flush=True)
        if n:
            frames.append(d)
    data = pd.concat(frames, ignore_index=True)
    data.to_parquet(f"{R}/data/processed/more_sites_table_fp.parquet")
    print(f"\nfootprint table: {len(data)} rows, {data.SITE_ID.nunique()} sites -> more_sites_table_fp.parquet")

    win = pd.read_parquet(f"{R}/data/processed/more_sites_table.parquet")
    print("\n=== leave-site-out R2 (ExtraTrees) — footprint vs 500 m window ===")
    for feats, name in [(M.FEATS, "all 14"), (PRUNED, "pruned 11")]:
        rw, mw = leave_site(win, feats)
        rf, mf = leave_site(data, feats)
        print(f"  {name:<10} window R2={rw:.3f} (MAE {mw:.2f}) | footprint R2={rf:.3f} (MAE {mf:.2f})")


if __name__ == "__main__":
    main()
