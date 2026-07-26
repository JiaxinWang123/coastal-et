"""Download gridMET daily meteorological forcing and extract per-tower series.

WHY gridMET AND NOT ERA5 (this was not the original plan):

  ERA5-Land via CDS is the higher-quality choice in principle (9 km, land-masked)
  but is unobtainable at this volume. CDS caps queued requests per dataset
  ("Number queued requests for this dataset is temporarily limited"), and the
  measured throughput was ~1 chunk / 18 min against 805 needed chunks -- roughly
  10 days of wall time.

  ARCO-ERA5 (the public zarr on GCS) sidesteps CDS entirely, but is chunked
  [1 time, 721 lat, 1440 lon] -- one full global field per hour. Extracting an
  11-year series at one tower would read ~96,000 global fields. Also infeasible.

  gridMET is 4 km (finer than either ERA5 product), CONUS, land-only, daily,
  1979-present, and is the forcing OpenET uses for US ET -- so this is the
  community-standard choice for exactly this problem, not a fallback.

  The cost is temporal: gridMET is DAILY, not hourly. That is fine for a daily-ET
  upscaling model (our target), but it cannot support instantaneous overpass-time
  energy balance. If TSEB/3SEB at overpass time is needed later, hourly forcing
  must be sourced separately (NLDAS-2 is the usual choice for CONUS).

Usage:  python download_gridmet.py [--extract-only]
Run via Slurm, never on the login node.
"""

import os
import sys
import time
import urllib.request
import concurrent.futures as cf

import numpy as np
import pandas as pd
import xarray as xr
import io_utils  # idempotent skip/overwrite helper (same src/ dir)

RAW = "/anvil/scratch/x-jwang120/coastal-et/data/raw/gridmet"
OUT = "/anvil/scratch/x-jwang120/coastal-et/data/interim/gridmet"
FULL = "/anvil/scratch/x-jwang120/coastal-et/data/processed/us_sites_coastal_distance.csv"
SITES = "/anvil/scratch/x-jwang120/coastal-et/config/coastal_sites.csv"

URL = "https://www.northwestknowledge.net/metdata/data/{var}_{year}.nc"
UA = "coastal-et/0.1 (research; Purdue Anvil)"

# gridMET internal variable names differ from the file names
VARS = {
    "tmmx": "air_temperature",                    # K, daily max
    "tmmn": "air_temperature",                    # K, daily min
    "rmax": "relative_humidity",                  # %, daily max
    "rmin": "relative_humidity",                  # %, daily min
    "vs": "wind_speed",                           # m/s at 10 m
    "srad": "surface_downwelling_shortwave_flux_in_air",  # W/m2, daily mean
    "pr": "precipitation_amount",                 # mm
    "etr": "potential_evapotranspiration",        # mm, alfalfa reference ET
    "pet": "potential_evapotranspiration",        # mm, grass reference ET
}
YEARS = range(2015, 2026)


def fetch(var, year):
    dest = os.path.join(RAW, f"{var}_{year}.nc")
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000 and not io_utils.force_requested():
        return dest, True   # already downloaded (pass --force to redownload)
    url = URL.format(var=var, year=year)
    tmp = dest + ".tmp"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=900) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
    os.rename(tmp, dest)
    return dest, False


def download_all():
    os.makedirs(RAW, exist_ok=True)
    jobs = [(v, y) for v in VARS for y in YEARS]
    print(f"downloading {len(jobs)} gridMET var-year files (~150 MB each)")

    def one(j):
        v, y = j
        for attempt in range(1, 4):
            try:
                _, cached = fetch(v, y)
                return j, cached, None
            except Exception as e:
                if attempt == 3:
                    return j, False, str(e)[:70]
                time.sleep(10 * attempt)

    ok = cached = 0
    bad = []
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        for (v, y), was_cached, err in ex.map(one, jobs):
            if err:
                bad.append(f"{v}_{y}: {err}")
            elif was_cached:
                cached += 1
            else:
                ok += 1
                if ok % 10 == 0:
                    print(f"  {ok + cached}/{len(jobs)}", flush=True)
    print(f"downloaded {ok}, cached {cached}, failed {len(bad)}")
    if bad:
        print("FAILED:")
        for b in bad:
            print("  " + b)
        sys.exit(1)


def es_kpa(T_C):
    """Saturation vapour pressure, kPa (Tetens)."""
    return 0.6108 * np.exp(17.27 * T_C / (T_C + 237.3))


def extract():
    os.makedirs(OUT, exist_ok=True)
    coords = pd.read_csv(FULL)[["SITE_ID", "LAT", "LON"]]
    sites = pd.read_csv(SITES).query("USE != 'exclude'").merge(coords, on="SITE_ID")

    # open each variable once across all years, then pull every tower from it --
    # far cheaper than reopening the grid per site
    series = {}
    for var in VARS:
        files = [os.path.join(RAW, f"{var}_{y}.nc") for y in YEARS]
        files = [f for f in files if os.path.exists(f)]
        if not files:
            print(f"  {var}: no files")
            continue
        ds = xr.open_mfdataset(files, combine="by_coords", parallel=False)
        name = [v for v in ds.data_vars][0]
        da = ds[name]
        lats = xr.DataArray(sites.LAT.values, dims="site")
        lons = xr.DataArray(sites.LON.values, dims="site")
        pt = da.sel(lat=lats, lon=lons, method="nearest").compute()
        series[var] = pd.DataFrame(
            pt.values, index=pd.to_datetime(pt["day"].values),
            columns=sites.SITE_ID.values)
        print(f"  {var}: {pt.shape[0]} days x {pt.shape[1]} sites", flush=True)

    for sid in sites.SITE_ID:
        dest = os.path.join(OUT, f"{sid}_gridmet.parquet")
        if not io_utils.should_write(dest, label=f"gridMET for {sid}"):
            continue
        df = pd.DataFrame({v: series[v][sid] for v in series if sid in series[v]})
        df.index.name = "date"

        # FAO-56 daily VPD from the max/min temperature and humidity pairs.
        # Using mean T and mean RH instead biases VPD low, because es() is convex.
        if {"tmmx", "tmmn", "rmax", "rmin"} <= set(df.columns):
            tx, tn = df["tmmx"] - 273.15, df["tmmn"] - 273.15
            es = (es_kpa(tx) + es_kpa(tn)) / 2
            ea = (es_kpa(tn) * df["rmax"] / 100 + es_kpa(tx) * df["rmin"] / 100) / 2
            df["TA_C"] = (tx + tn) / 2
            df["VPD_kPa"] = (es - ea).clip(lower=0)
            df["RH_pct"] = (ea / es * 100).clip(0, 100)
        if "vs" in df:
            df["WS_ms"] = df["vs"]
        if "srad" in df:
            df["SRAD_Wm2"] = df["srad"]
        if "etr" in df:
            df["ETr_mm"] = df["etr"]     # alfalfa reference ET
        if "pet" in df:
            df["ETo_mm"] = df["pet"]     # grass reference ET
        if "pr" in df:
            df["P_mm"] = df["pr"]

        dest = os.path.join(OUT, f"{sid}_gridmet.parquet")
        df.to_parquet(dest)

    n = len(list(sites.SITE_ID))
    print(f"\nwrote {n} per-site gridMET series to {OUT}")

    # sanity: a marsh in Delaware and a marsh in Florida must not look identical
    chk = []
    for sid in sites.SITE_ID:
        d = pd.read_parquet(os.path.join(OUT, f"{sid}_gridmet.parquet"))
        if "TA_C" in d and len(d):
            chk.append(dict(site=sid, TA=round(d.TA_C.median(), 1),
                            VPD=round(d.VPD_kPa.median(), 2),
                            ETo=round(d.ETo_mm.median(), 2) if "ETo_mm" in d else np.nan,
                            n=len(d)))
    c = pd.DataFrame(chk)
    print(c.to_string(index=False))
    bad = c[(c.TA < -5) | (c.TA > 30) | (c.VPD < 0) | (c.VPD > 4)]
    if len(bad):
        raise ValueError(f"implausible gridMET forcing at: {list(bad.site)}")
    print("\nall sites within plausible ranges")


if __name__ == "__main__":
    if "--extract-only" not in sys.argv:
        download_all()
    extract()
