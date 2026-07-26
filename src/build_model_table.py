"""Assemble the final modelling table for the team's 5 Everglades sites, and
report honestly what is actually usable.

Target  : ET_closed_mm  (energy-balance-closed daily ET from the tower)
Features: Landsat LST + NDVI + MNDWI, Sentinel-2, ERA5 (or gridMET), seasonality

The join is where a dataset quietly dies, so every drop is counted:

  ET days                 -> tower has a closed daily ET
  + met                   -> ERA5 (FLUXNET AUX) or gridMET forcing exists
  + LST observed          -> a real cloud-free Landsat overpass THAT DAY
  + LST interpolated      -> LST filled within a 16-day gap (still usable, but
                             the model must be told how stale it is via *_age)

Two model scopes fall out, and they are NOT the same experiment:

  STRICT  rows with a real same-day LST  -- small, honest, thermally anchored
  FILLED  rows with interpolated LST     -- large, but LST is partly invented

Report both. A model trained on FILLED and validated on FILLED will look far
better than it is, because interpolated LST carries the autocorrelation of the
ET it is meant to predict.

Run via Slurm, never on the login node.
"""

import os
import glob

import numpy as np
import pandas as pd

R = "/anvil/scratch/x-jwang120/coastal-et"
OUT = f"{R}/data/processed"
FLXNET = f"{R}/data/raw/ameriflux/fluxnet"

TEAM = ["US-Skr", "US-Elm", "US-Esm", "US-EvM", "US-TaS"]
STRATUM = {"US-Skr": "MANGROVE", "US-EvM": "SALT_INTRUSION",
           "US-TaS": "TAYLOR_SLOUGH", "US-Esm": "FRESH_SHORT_HYDRO",
           "US-Elm": "FRESH_LONG_HYDRO"}

MAX_GAP = 16          # Landsat repeat cycle; filling past it is invention
SAT_VARS = ["LST_K", "NDVI", "MNDWI"]


def era5_daily(site):
    f = glob.glob(f"{FLXNET}/{site}/**/*ERA5_DD*.csv", recursive=True)
    if not f:
        return None
    e = pd.read_csv(f[0], na_values=[-9999, "-9999"], low_memory=False)
    e.index = pd.to_datetime(e["TIMESTAMP"].astype(str), format="%Y%m%d", utc=True)
    keep = [c for c in ["TA_ERA", "SW_IN_ERA", "LW_IN_ERA", "VPD_ERA",
                        "PA_ERA", "P_ERA", "WS_ERA"] if c in e.columns]
    d = e[keep].copy()
    d["MET_SOURCE"] = "ERA5_FLUXNET"
    return d


def gridmet_daily(site):
    p = f"{R}/data/interim/gridmet/{site}_gridmet.parquet"
    if not os.path.exists(p):
        return None
    g = pd.read_parquet(p)
    g.index = pd.to_datetime(g.index, utc=True)
    # rename to the ERA5 names so the feature matrix is uniform across sites
    ren = {"TA_C": "TA_ERA", "SRAD_Wm2": "SW_IN_ERA", "VPD_kPa": "VPD_ERA",
           "WS_ms": "WS_ERA", "P_mm": "P_ERA"}
    g = g[[c for c in ren if c in g]].rename(columns=ren)
    g["MET_SOURCE"] = "gridMET"
    return g


def satellite_daily(site):
    p = f"{R}/data/interim/satellite/{site}_satellite.parquet"
    if not os.path.exists(p):
        return None
    s = pd.read_parquet(p)
    s.index = pd.to_datetime(s.index, utc=True)
    feats = [c for c in SAT_VARS if c in s.columns]
    d = s[feats].groupby(s.index.normalize()).mean()

    out = pd.DataFrame(index=d.index)
    for c in feats:
        out[f"{c}_obs"] = d[c]
    return out


def main():
    et = pd.read_parquet(f"{OUT}/daily_closed_et.parquet")
    et.index = pd.to_datetime(et.index, utc=True)

    frames, rep = [], []
    for s in TEAM:
        e = et[(et.SITE_ID == s) & (et.ET_closed_mm.notna())].copy()
        n_et = len(e)
        if not n_et:
            rep.append(dict(site=s, ET=0)); continue

        met = era5_daily(s)
        if met is None:
            met = gridmet_daily(s)
        sat = satellite_daily(s)

        df = e[["ET_closed_mm", "ET_open_mm", "CLOSURE"]].copy()
        for c in ["ET_lo_mm", "ET_hi_mm", "ET_randunc_mm"]:
            if c in e:
                df[c] = e[c]

        n_met = 0
        if met is not None:
            df = df.join(met, how="left")
            n_met = int(df["TA_ERA"].notna().sum()) if "TA_ERA" in df else 0

        n_obs = 0
        if sat is not None:
            # reindex onto a CONTINUOUS daily axis before interpolating, or gaps
            # get silently collapsed and *_age becomes meaningless
            full = pd.date_range(df.index.min(), df.index.max(), freq="D", tz="UTC")
            sat = sat.reindex(full)
            for c in SAT_VARS:
                o = sat.get(f"{c}_obs")
                if o is None:
                    continue
                sat[f"{c}_int"] = o.interpolate(limit=MAX_GAP, limit_area="inside")
                idx = pd.Series(np.arange(len(o)), index=o.index)
                sat[f"{c}_age"] = (idx - idx.where(o.notna()).ffill()).astype(float)
            df = df.join(sat, how="left")
            n_obs = int(df["LST_K_obs"].notna().sum()) if "LST_K_obs" in df else 0

        df["SITE_ID"] = s
        df["STRATUM"] = STRATUM[s]
        doy = df.index.dayofyear
        df["DOY_sin"] = np.sin(2 * np.pi * doy / 365.25)
        df["DOY_cos"] = np.cos(2 * np.pi * doy / 365.25)

        # REQUIRE the met columns. Filtering on "columns that happen to exist"
        # silently passes a site with no forcing at all, which is how US-Skr
        # first showed up as 43 usable rows while having zero meteorology.
        need = ["ET_closed_mm", "TA_ERA", "VPD_ERA", "SW_IN_ERA"]
        missing_cols = [c for c in need if c not in df.columns]
        if missing_cols:
            print(f"  {s}: MISSING FORCING {missing_cols} -> 0 usable rows")
            have_met = pd.Series(False, index=df.index)
        else:
            have_met = df[need].notna().all(axis=1)
        strict = have_met & df.get("LST_K_obs", pd.Series(False, index=df.index)).notna()
        filled = have_met & df.get("LST_K_int", pd.Series(False, index=df.index)).notna()

        frames.append(df)
        rep.append(dict(
            site=s, stratum=STRATUM[s], ET=n_et,
            met=n_met, met_src=(met["MET_SOURCE"].iloc[0] if met is not None else "-"),
            LST_obs=n_obs,
            STRICT=int(strict.sum()), FILLED=int(filled.sum()),
            strict_22_23=int(strict[(strict.index >= "2022-01-01") &
                                    (strict.index <= "2023-12-31")].sum()),
        ))

    data = pd.concat(frames).sort_values(["SITE_ID"], kind="stable")
    data.index.name = "date"
    data.to_parquet(f"{OUT}/model_table_everglades.parquet")
    data.to_csv(f"{OUT}/model_table_everglades.csv")

    r = pd.DataFrame(rep)
    print("=== MODELLING TABLE: what survives the join? ===\n")
    print(r.to_string(index=False))
    print(f"\n  STRICT = real same-day Landsat LST  (thermally anchored, honest)")
    print(f"  FILLED = LST interpolated within {MAX_GAP} d  (bigger, but LST partly invented)")
    print(f"\n  TOTAL STRICT rows : {r.STRICT.sum():,}   <-- the real training set")
    print(f"  TOTAL FILLED rows : {r.FILLED.sum():,}")
    print(f"  STRICT in 2022-23 : {r.strict_22_23.sum():,}")

    feats = [c for c in data.columns if c.endswith(("_obs", "_int", "_age"))
             or c in ("TA_ERA", "VPD_ERA", "SW_IN_ERA", "LW_IN_ERA", "WS_ERA",
                      "P_ERA", "PA_ERA", "DOY_sin", "DOY_cos")]
    print(f"\n  features available: {len(feats)}")
    print(f"  {sorted(feats)}")

    print("\n=== can we actually validate? ===")
    print(f"  sites: {r[r.STRICT>0].shape[0]}  -> leave-one-SITE-out CV is possible")
    thin = r[(r.STRICT > 0) & (r.STRICT < 50)]
    if len(thin):
        print(f"  WARNING: sites with <50 strict rows (a fold will be meaningless):")
        for _, x in thin.iterrows():
            print(f"    {x.site}: {x.STRICT} rows")
    print(f"\nwrote {OUT}/model_table_everglades.parquet (+ .csv)")


if __name__ == "__main__":
    main()
