"""Join flux, satellite, gridMET and tide streams into the ML-ready daily table.

The four streams live on different clocks, and how you reconcile them is a
modelling decision, not a formatting one:

  flux      daily     the reference ET (target)
  gridMET   daily     meteorological forcing -- aligns directly
  satellite irregular Landsat every ~8-16 d, Sentinel-2 ~5 d, Sentinel-1 ~6-12 d
  tide      30-min    aggregated to daily inundation statistics

Satellite is the hard one. A daily ET model needs a value every day, but the
sensors only see the site every few days -- and interpolating a cloud-free LST
across a 16-day gap invents data. So we do both, and label which is which:

  *_obs   the actual observation, NaN on days with no overpass
  *_int   gap-filled by time interpolation, with a hard limit on gap length
  *_age   days since the nearest real observation -- the honesty column

`*_age` matters: a model trained on `*_int` without knowing the observation is
14 days stale will happily trust a stale LST. Feeding `*_age` in lets it learn to
discount them, and lets you filter on it at evaluation time.

Usage:  python build_dataset.py
Run via Slurm, never on the login node.
"""

import os
import glob

import numpy as np
import pandas as pd

ROOT = "/anvil/scratch/x-jwang120/coastal-et"
FLUX = f"{ROOT}/data/interim/flux"
SAT = f"{ROOT}/data/interim/satellite"
MET = f"{ROOT}/data/interim/gridmet"
TIDE = f"{ROOT}/data/raw/tides"
OUT = f"{ROOT}/data/processed"
SITES = f"{ROOT}/config/coastal_sites.csv"

MAX_GAP_DAYS = 16   # Landsat's repeat cycle; interpolating past it is invention

SAT_FEATURES = ["LST_K", "NDVI", "MNDWI", "S1_VV_dB", "S1_VH_dB", "S1_RATIO",
                "landsat_lwir11_std", "landsat_red_std"]


def daily_tide(sid):
    p = os.path.join(TIDE, f"{sid}_tide.parquet")
    if not os.path.exists(p):
        return None
    t = pd.read_parquet(p)
    t.index = pd.to_datetime(t.index, utc=True)
    d = t.resample("D").agg({
        "water_level_m": ["mean", "max", "min"],
        "inundation_frac_24h": "mean",
        "wl_rate_m_per_hr": "std",
    })
    d.columns = ["WL_mean_m", "WL_max_m", "WL_min_m", "INUND_frac", "WL_rate_std"]
    d["WL_range_m"] = d["WL_max_m"] - d["WL_min_m"]   # tidal amplitude that day
    return d


def daily_satellite(sid):
    p = os.path.join(SAT, f"{sid}_satellite.parquet")
    if not os.path.exists(p):
        return None
    s = pd.read_parquet(p)
    s.index = pd.to_datetime(s.index, utc=True)
    keep = [c for c in SAT_FEATURES if c in s.columns]
    s = s[keep]
    # several sensors can see the site on one day; average them
    d = s.resample("D").mean()

    out = pd.DataFrame(index=d.index)
    for c in keep:
        obs = d[c]
        out[f"{c}_obs"] = obs
        out[f"{c}_int"] = obs.interpolate(limit=MAX_GAP_DAYS, limit_area="inside")
        # days since the last real observation of this variable
        idx = pd.Series(np.arange(len(obs)), index=obs.index)
        last = idx.where(obs.notna()).ffill()
        out[f"{c}_age"] = (idx - last).astype(float)
    return out


import io_utils  # idempotent skip/overwrite helper (same src/ dir)


def main():
    os.makedirs(OUT, exist_ok=True)
    if not io_utils.should_write(os.path.join(OUT, "coastal_et_dataset.parquet"),
                                 label="coastal_et_dataset"):
        return   # already built
    meta = pd.read_csv(SITES).set_index("SITE_ID")

    frames, report = [], []
    for f in sorted(glob.glob(os.path.join(FLUX, "*_daily.parquet"))):
        sid = os.path.basename(f).split("_")[0]

        flux = pd.read_parquet(f)
        flux.index = pd.to_datetime(flux.index, utc=True)
        flux = flux[flux["ET_corr_mm"].notna()]
        if flux.empty:
            continue

        df = flux[["ET_corr_mm", "ET_raw_mm", "NETRAD_Wm2", "TA_C", "WTD_m",
                   "coverage"]].copy()
        df = df.rename(columns={"NETRAD_Wm2": "NETRAD_tower", "TA_C": "TA_tower"})

        met_p = os.path.join(MET, f"{sid}_gridmet.parquet")
        if os.path.exists(met_p):
            m = pd.read_parquet(met_p)
            m.index = pd.to_datetime(m.index, utc=True)
            keep = [c for c in ["TA_C", "VPD_kPa", "RH_pct", "WS_ms", "SRAD_Wm2",
                                "ETr_mm", "ETo_mm", "P_mm"] if c in m.columns]
            df = df.join(m[keep], how="left")

        sat = daily_satellite(sid)
        if sat is not None:
            df = df.join(sat, how="left")

        tid = daily_tide(sid)
        if tid is not None:
            df = df.join(tid, how="left")

        # antecedent memory: marsh ET depends on recent wetness and inundation
        # history, which same-day features cannot express
        if "P_mm" in df:
            df["P_7d"] = df["P_mm"].rolling(7, min_periods=1).sum()
            df["P_30d"] = df["P_mm"].rolling(30, min_periods=1).sum()
        if "INUND_frac" in df:
            df["INUND_7d"] = df["INUND_frac"].rolling(7, min_periods=1).mean()
        if "ETo_mm" in df:
            # evaporative fraction against reference ET: the classic scaling ratio
            df["EF_ratio"] = df["ET_corr_mm"] / df["ETo_mm"].replace(0, np.nan)

        df["DOY"] = df.index.dayofyear
        df["DOY_sin"] = np.sin(2 * np.pi * df["DOY"] / 365.25)
        df["DOY_cos"] = np.cos(2 * np.pi * df["DOY"] / 365.25)

        df["SITE_ID"] = sid
        df["STRATUM"] = meta.loc[sid, "STRATUM"] if sid in meta.index else "?"
        df["USE"] = meta.loc[sid, "USE"] if sid in meta.index else "?"
        # estuary grouping: co-located towers must never be split across a
        # train/test boundary, or the model leaks through shared meteorology
        df["ESTUARY"] = ESTUARY.get(sid, sid)

        frames.append(df)
        report.append(dict(
            site=sid, days=len(df), stratum=df["STRATUM"].iloc[0],
            has_sat=sat is not None, has_tide=tid is not None,
            lst_obs=int(df["LST_K_obs"].notna().sum()) if "LST_K_obs" in df else 0,
            lst_cov=round(100 * df["LST_K_int"].notna().mean(), 0) if "LST_K_int" in df else 0,
        ))

    data = pd.concat(frames).sort_values(["SITE_ID", "coverage"])
    data.index.name = "date"
    data.to_parquet(os.path.join(OUT, "coastal_et_dataset.parquet"))

    r = pd.DataFrame(report).sort_values("days", ascending=False)
    print(r.to_string(index=False))
    print(f"\nrows (site-days)      : {len(data):,}")
    print(f"sites                 : {data.SITE_ID.nunique()}")
    print(f"independent estuaries : {data.ESTUARY.nunique()}   <- the number that "
          "governs model complexity")
    print(f"strata                : {sorted(data.STRATUM.unique())}")
    print(f"features              : {len(data.columns)}")

    print("\nrows usable with a REAL (not interpolated) LST observation:")
    if "LST_K_obs" in data:
        n = data["LST_K_obs"].notna().sum()
        print(f"  {n:,} / {len(data):,}  ({100*n/len(data):.0f}%)")
    print("rows with interpolated LST within a 16-day gap:")
    if "LST_K_int" in data:
        n = data["LST_K_int"].notna().sum()
        print(f"  {n:,} / {len(data):,}  ({100*n/len(data):.0f}%)")

    print(f"\nwrote {OUT}/coastal_et_dataset.parquet")


# Co-located towers share an estuary and therefore share weather, tides and
# often instrumentation. Splitting them across train/test leaks. This mapping is
# what leave-one-ESTUARY-out cross-validation groups on.
ESTUARY = {
    "US-EKH": "Elkhorn", "US-EKY": "Elkhorn", "US-EKP": "Elkhorn",
    "US-EKN": "Elkhorn", "US-MCP": "Elkhorn",
    "US-PHM": "PlumIsland", "US-PLM": "PlumIsland", "US-PLo": "PlumIsland",
    "US-HRP": "HerringRiver", "US-HRo": "HerringRiver",
    "US-HB1": "NorthInlet", "US-HB4": "NorthInlet", "US-HB7": "NorthInlet",
    "US-StS": "NorthInlet",
    "US-Tw1": "Delta", "US-Tw4": "Delta", "US-Tw5": "Delta",
    "US-Myb": "Delta", "US-Dmg": "Delta", "US-Srr": "Delta",
    "US-KS3": "KennedySpace", "US-KS4": "KennedySpace",
    "US-MRM": "Meadowlands", "US-SHS": "Meadowlands", "US-HPY": "Meadowlands",
    "US-EvM": "Everglades", "US-TaS": "Everglades",
    "US-SJ1": "SanJoaquin", "US-SJ2": "SanJoaquin",
    "US-Nrs": "Nisqually", "US-Nrf": "Nisqually",
}


if __name__ == "__main__":
    main()
