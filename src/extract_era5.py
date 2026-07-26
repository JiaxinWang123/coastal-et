"""Extract per-tower ERA5-Land forcing from the downloaded regional boxes.

Two traps handled here, both of which corrupt the forcing silently:

1. ERA5-Land is MASKED OVER OCEAN. A marsh tower 100 m from open water can land
   in a cell ERA5-Land calls sea, giving all-NaN. We take the nearest cell with
   real land data and record how far away it was, so a tower being forced by
   inland meteorology is visible rather than hidden.

2. ERA5-Land accumulated fluxes (radiation, precip) ACCUMULATE FROM 00 UTC and
   reset daily. Read as instantaneous, late-afternoon Rn is inflated by ~an
   order of magnitude. We difference within each UTC day to recover hourly rates.

Usage:  python extract_era5.py SITE_ID
Run via Slurm, never on the login node.
"""

import os
import sys
import glob

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, "/anvil/scratch/x-jwang120/coastal-et/config")
from era5_regions import REGIONS  # noqa: E402

RAW = "/anvil/scratch/x-jwang120/coastal-et/data/raw/era5"
OUT = "/anvil/scratch/x-jwang120/coastal-et/data/interim/era5"
FULL = "/anvil/scratch/x-jwang120/coastal-et/data/processed/us_sites_coastal_distance.csv"

ACCUM_SHORT = ["ssr", "str", "ssrd", "tp"]


def region_of(site_id):
    for name, cfg in REGIONS.items():
        if site_id in cfg["sites"]:
            return name
    raise SystemExit(f"{site_id} is in no ERA5 region")


def nearest_land_cell(ds, lat, lon):
    probe = ds["t2m"].isel(valid_time=slice(0, 48)).mean("valid_time").compute()
    valid = probe.notnull().values
    if not valid.any():
        raise SystemExit("no valid ERA5-Land cell in the box (entirely sea?)")

    la = ds["latitude"].values
    lo = ds["longitude"].values
    LO, LA = np.meshgrid(lo, la)
    d = np.sqrt(((LA - lat) * 111.32) ** 2
                + ((LO - lon) * 111.32 * np.cos(np.radians(lat))) ** 2)
    d = np.where(valid, d, np.inf)
    j, i = np.unravel_index(np.argmin(d), d.shape)
    return float(la[j]), float(lo[i]), float(d[j, i])


def deaccumulate(df):
    """ERA5-Land fluxes are cumulative from 00 UTC and reset daily -> difference."""
    for v in [c for c in ACCUM_SHORT if c in df]:
        s = df[v]
        day = s.index.date
        d = s.groupby(day).diff()
        first = s.groupby(day).transform("first")
        df[v] = d.where(d.notna(), first)   # 01:00 already holds hour 1's total
    for v in ["ssr", "str", "ssrd"]:
        if v in df:
            df[v] = df[v] / 3600.0          # J m-2 per hour -> W m-2
    if "tp" in df:
        df["tp"] = df["tp"] * 1000.0        # m -> mm
    return df


def derive(df):
    if "t2m" in df and "d2m" in df:
        Ta = df["t2m"] - 273.15
        Td = df["d2m"] - 273.15
        es = 0.6108 * np.exp(17.27 * Ta / (Ta + 237.3))     # kPa, Tetens
        ea = 0.6108 * np.exp(17.27 * Td / (Td + 237.3))
        df["TA_C"] = Ta
        df["VPD_kPa"] = (es - ea).clip(lower=0)
        df["RH_pct"] = (ea / es * 100).clip(0, 100)
    if "u10" in df and "v10" in df:
        df["WS_ms"] = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)
    if "ssr" in df and "str" in df:
        df["NETRAD_Wm2"] = df["ssr"] + df["str"]
    if "swvl1" in df:
        df["SWC_m3m3"] = df["swvl1"]
    if "skt" in df:
        df["SKT_K"] = df["skt"]
    return df


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    site_id = sys.argv[1]
    region = region_of(site_id)

    row = pd.read_csv(FULL).query("SITE_ID == @site_id")
    if row.empty:
        raise SystemExit(f"{site_id}: no coordinates")
    lat, lon = float(row.iloc[0].LAT), float(row.iloc[0].LON)

    files = sorted(glob.glob(os.path.join(RAW, f"{region}_*.nc")))
    if not files:
        raise SystemExit(f"no ERA5 files for region {region} -- run download_era5.py")
    print(f"{site_id}  region={region}  ({lat:.4f}, {lon:.4f})  {len(files)} chunks")

    ds = xr.open_mfdataset(files, combine="by_coords", parallel=False)
    clat, clon, dist = nearest_land_cell(ds, lat, lon)
    print(f"  nearest ERA5-Land LAND cell: ({clat:.2f}, {clon:.2f})  {dist:.1f} km away")
    if dist > 15:
        print("  WARNING: >15 km. This tower sits in a sea-masked area, so its")
        print("           forcing comes from inland and may misrepresent the marsh.")

    pt = ds.sel(latitude=clat, longitude=clon, method="nearest")
    df = pt.to_dataframe()
    df = df[[c for c in df.columns if c not in ("latitude", "longitude", "number",
                                                "expver")]]
    df.index = pd.to_datetime(df.index.get_level_values("valid_time"), utc=True)
    df = df[~df.index.duplicated(keep="first")].sort_index()

    df = deaccumulate(df)
    df = derive(df)

    # loud gates: these catch de-accumulation and unit errors instead of letting
    # a plausible-looking but wrong forcing series into the model
    if "NETRAD_Wm2" in df:
        rn = df["NETRAD_Wm2"].dropna()
        if len(rn) and rn.max() > 1400:
            raise ValueError(f"peak Rn {rn.max():.0f} W/m2 exceeds the solar "
                             "constant -- de-accumulation is wrong.")
    if "TA_C" in df:
        ta = df["TA_C"].dropna()
        if len(ta) and not (-30 < ta.median() < 40):
            raise ValueError(f"median TA {ta.median():.1f} C is implausible.")

    os.makedirs(OUT, exist_ok=True)
    dest = os.path.join(OUT, f"{site_id}_era5land.parquet")
    df.attrs = {}
    df.to_parquet(dest)

    print(f"wrote {dest}  ({len(df)} hours, {len(df.columns)} vars)")
    for c, unit in [("TA_C", "C"), ("VPD_kPa", "kPa"), ("NETRAD_Wm2", "W/m2"),
                    ("WS_ms", "m/s")]:
        if c in df:
            s = df[c].dropna()
            print(f"  {c:<11} median={s.median():8.2f}  max={s.max():8.2f} {unit}")


if __name__ == "__main__":
    main()
