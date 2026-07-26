"""Footprint-weighted pixel-level ET model + honest validation.

Each 30 m pixel is a training sample: per-pixel satellite (LST/NDVI/MNDWI) +
per-date met/ETo/season -> tower ET, weighted by the pixel's Kljun-2015 footprint
contribution. Pooling all 5 sites fixes the extrapolation ceiling.

VALIDATION (both, to expose the leakage the user was warned about):
  WITHIN-TOWER  random split of tower-dates      -- optimistic (shared ET label)
  LEAVE-TOWER-OUT  train 4 sites, predict 5th     -- honest spatial-scaling test
For each, the pixel predictions are footprint-averaged per date and compared to
the measured tower ET (the operationally meaningful metric for a map).
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
SATF = ["LST_K", "NDVI", "MNDWI"]
METF = ["TA_ERA", "VPD_ERA", "SW_IN_ERA", "WS_ERA", "ETo_mm", "DOY_sin", "DOY_cos"]
FEATS = SATF + METF


def load():
    et = pd.read_parquet(f"{OUT}/daily_closed_et.parquet")
    et.index = pd.to_datetime(et.index, utc=True)
    frames = []
    for s in SITES:
        p = f"{PIX}/{s}_pixels.parquet"
        if not os.path.exists(p):
            continue
        px = pd.read_parquet(p)
        px["date"] = pd.to_datetime(px["date"], utc=True)
        e = et[(et.SITE_ID == s) & (et.ET_closed_mm.notna())][["ET_closed_mm"]]
        e.index = e.index.normalize()
        m = G.merged_met(s)
        gm = f"{R}/data/interim/gridmet/{s}_gridmet.parquet"
        eto = None
        if os.path.exists(gm):
            g = pd.read_parquet(gm)
            g.index = pd.to_datetime(g.index, utc=True)
            if "ETo_mm" in g:
                eto = g[["ETo_mm"]]
        perdate = e.join(m, how="left")
        if eto is not None:
            perdate = perdate.join(eto, how="left")
        perdate.index = perdate.index.normalize()
        px = px.merge(perdate, left_on="date", right_index=True, how="inner")
        doy = px["date"].dt.dayofyear
        px["DOY_sin"] = np.sin(2 * np.pi * doy / 365.25)
        px["DOY_cos"] = np.cos(2 * np.pi * doy / 365.25)
        frames.append(px)
    return pd.concat(frames, ignore_index=True)


def rf():
    return RandomForestRegressor(300, min_samples_leaf=5, random_state=42, n_jobs=-1)


def fp_avg(df, pred):
    """footprint-weighted mean prediction per site-date -> compare to tower ET."""
    df = df.assign(p=pred)
    g = df.groupby(["SITE_ID", "date"]).apply(
        lambda x: pd.Series({"pred": np.average(x.p, weights=x.fp_weight),
                             "obs": x.ET_closed_mm.iloc[0]}))
    return g.reset_index()


def main():
    d = load().dropna(subset=["ET_closed_mm"] + FEATS)
    print(f"{len(d):,} pixel-obs, {d.SITE_ID.nunique()} sites, "
          f"{d.groupby('SITE_ID').date.nunique().to_dict()}\n")

    # WITHIN-TOWER: random 70/30 split of (site,date) blocks -- no date in both
    keys = d[["SITE_ID", "date"]].drop_duplicates().sample(frac=1, random_state=1)
    cut = int(0.7 * len(keys))
    trk, tek = keys.iloc[:cut], keys.iloc[cut:]
    tr = d.merge(trk, on=["SITE_ID", "date"])
    te = d.merge(tek, on=["SITE_ID", "date"])
    m = rf()
    m.fit(tr[FEATS].values, tr.ET_closed_mm.values, sample_weight=tr.fp_weight.values)
    g = fp_avg(te, m.predict(te[FEATS].values))
    print("=== WITHIN-TOWER (random date split; footprint-avg per date) ===")
    print(f"  R2={r2_score(g.obs, g.pred):.3f}  MAE={mean_absolute_error(g.obs, g.pred):.2f}"
          f"  n_dates={len(g)}")

    # LEAVE-TOWER-OUT
    print("\n=== LEAVE-TOWER-OUT (honest spatial scaling) ===")
    allg = []
    for s in SITES:
        tr = d[d.SITE_ID != s]
        te = d[d.SITE_ID == s]
        if len(te) < 50:
            continue
        m = rf()
        m.fit(tr[FEATS].values, tr.ET_closed_mm.values, sample_weight=tr.fp_weight.values)
        g = fp_avg(te, m.predict(te[FEATS].values))
        allg.append(g)
        print(f"  {s}: R2={r2_score(g.obs, g.pred):>6.3f}  "
              f"MAE={mean_absolute_error(g.obs, g.pred):.2f}  n={len(g)}")
    G2 = pd.concat(allg)
    print(f"  POOLED: R2={r2_score(G2.obs, G2.pred):.3f}  MAE={mean_absolute_error(G2.obs, G2.pred):.2f}")
    G2.to_parquet(f"{OUT}/pixel_loto_predictions.parquet")

    # final model on ALL data -> for the map
    m = rf()
    m.fit(d[FEATS].values, d.ET_closed_mm.values, sample_weight=d.fp_weight.values)
    import joblib
    joblib.dump({"model": m, "feats": FEATS}, f"{OUT}/pixel_et_model.joblib")
    imp = pd.Series(m.feature_importances_, index=FEATS).sort_values(ascending=False)
    print("\n  feature importance:", ", ".join(f"{k}={v:.2f}" for k, v in imp.head(6).items()))
    print(f"  saved model -> {OUT}/pixel_et_model.joblib")


if __name__ == "__main__":
    main()
