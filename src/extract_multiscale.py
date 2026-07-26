"""Extract satellite predictors at MULTIPLE radii around each tower.

Why: a single chip size is a guess about the eddy-covariance footprint, and the
wrong guess quietly destroys the signal. The footprint scales roughly as
100 x (z_m - d):

    US-Skr  mangrove, z_m~27 m, h_c~20 m  ->  ~1.4 km
    marshes z_m~4 m,   h_c~1 m            ->  ~0.3 km

A 6 km chip (what we used first) averages 19x too much area at the mangrove and
325x too much at the marshes -- folding the Shark River channel and open bay into
a number meant to represent a tower's fetch.

Rather than swap one guess for another, we extract concentric radii and let the
model tell us which scale predicts tower ET best. That is an empirical answer to
the footprint question, and the scale-vs-skill curve is itself a result worth
reporting.

NOTE on thermal: Landsat TIRS senses at 100 m and is resampled to 30 m, so radii
below ~100 m carry no independent thermal information -- they are included for the
optical bands and as a check, not because LST truly resolves them.

Usage:  python extract_multiscale.py SITE_ID [START] [END]
Run via Slurm, never on the login node.
"""

import os
import sys
import time

os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "6")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "3")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("VSI_CACHE", "TRUE")

import numpy as np
import pandas as pd
import dask
import planetary_computer as pc
import pystac_client
import stackstac

dask.config.set(scheduler="threads", num_workers=32)

OUT = "/anvil/scratch/x-jwang120/coastal-et/data/interim/multiscale"
FULL = "/anvil/scratch/x-jwang120/coastal-et/data/processed/us_sites_coastal_distance.csv"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"

# metres from the tower. 100 m is the native Landsat thermal cell; 2000 m is
# beyond any plausible footprint here and acts as the "too big" control.
RADII = [100, 250, 500, 1000, 2000]
OUTER_KM = 2.2          # download once at the largest radius, then sub-aggregate
MAX_TRIES = 3

COLLECTIONS = {
    "landsat": dict(name="landsat-c2-l2", res=30,
                    assets=["red", "nir08", "green", "swir16", "lwir11", "qa_pixel"]),
    "sentinel2": dict(name="sentinel-2-l2a", res=20,
                      assets=["B03", "B04", "B08", "B11", "SCL"]),
}


def coords(site_id):
    r = pd.read_csv(FULL).query("SITE_ID == @site_id")
    if r.empty:
        raise SystemExit(f"{site_id}: no coordinates")
    return float(r.iloc[0].LAT), float(r.iloc[0].LON)


def qa_mask(da, coll):
    bands = list(da.band.values)
    if coll == "landsat" and "qa_pixel" in bands:
        qa = da.sel(band="qa_pixel").fillna(1).astype("uint16")
        bad = (qa & 0b111110) > 0     # dilated cloud, cirrus, cloud, shadow, snow
        keep = [b for b in bands if b != "qa_pixel"]
        return da.sel(band=keep).where(~bad)
    if coll == "sentinel2" and "SCL" in bands:
        scl = da.sel(band="SCL")
        good = scl.isin([4, 5, 6, 7])
        keep = [b for b in bands if b != "SCL"]
        return da.sel(band=keep).where(good)
    return da


def fetch(coll, bbox, start, end):
    cfg = COLLECTIONS[coll]
    cat = pystac_client.Client.open(STAC, modifier=pc.sign_inplace)
    items = list(cat.search(collections=[cfg["name"]], bbox=bbox,
                            datetime=f"{start}/{end}").items())
    if not items:
        return None
    return stackstac.stack(items, assets=cfg["assets"], bounds_latlon=bbox,
                           epsg=3857, resolution=cfg["res"], chunksize=512)


def radial_means(da, coll, lat, lon):
    """Mean of each band within each radius of the tower, in one dask pass."""
    da = qa_mask(da, coll)

    # tower position in the chip's projected CRS (EPSG:3857)
    import pyproj
    tx, ty = pyproj.Transformer.from_crs(4326, 3857, always_xy=True).transform(lon, lat)
    X, Y = np.meshgrid(da.x.values, da.y.values)
    # EPSG:3857 inflates distance by 1/cos(lat); correct it or radii are wrong
    dist = np.sqrt((X - tx) ** 2 + (Y - ty) ** 2) * np.cos(np.radians(lat))

    tasks, keys = [], []
    for r in RADII:
        m = xr_mask(da, dist <= r)
        tasks.append(m.mean(dim=("y", "x"), skipna=True))
        keys.append(r)
    results = dask.compute(*tasks)

    frames = []
    for r, res in zip(keys, results):
        df = res.to_pandas()
        df.columns = [f"{coll}_{b}_r{r}" for b in df.columns]
        frames.append(df)
    out = pd.concat(frames, axis=1)
    out.index.name = "time"
    return out.groupby(level=0).mean().sort_index().dropna(how="all")


def xr_mask(da, mask2d):
    import xarray as xr
    m = xr.DataArray(mask2d, dims=("y", "x"),
                     coords={"y": da.y, "x": da.x})
    return da.where(m)


def add_indices(df):
    for r in RADII:
        red, nir = f"landsat_red_r{r}", f"landsat_nir08_r{r}"
        grn, swir = f"landsat_green_r{r}", f"landsat_swir16_r{r}"
        lwir = f"landsat_lwir11_r{r}"
        if red in df and nir in df:
            df[f"NDVI_r{r}"] = (df[nir] - df[red]) / (df[nir] + df[red])
        if grn in df and swir in df:
            df[f"MNDWI_r{r}"] = (df[grn] - df[swir]) / (df[grn] + df[swir])
        if lwir in df:
            df[f"LST_K_r{r}"] = df[lwir]     # already kelvin (stackstac rescales)
        b8, b4 = f"sentinel2_B08_r{r}", f"sentinel2_B04_r{r}"
        if b8 in df and b4 in df:
            df[f"NDVI_S2_r{r}"] = (df[b8] - df[b4]) / (df[b8] + df[b4])
    return df


import io_utils  # idempotent skip/overwrite helper (same src/ dir)


def main():
    argv = io_utils.clean_argv()
    site = argv[1]
    start = argv[2] if len(argv) > 2 else "2022-01-01"
    end = argv[3] if len(argv) > 3 else "2023-12-31"

    os.makedirs(OUT, exist_ok=True)
    dest = os.path.join(OUT, f"{site}_multiscale.parquet")
    if not io_utils.should_write(dest, label=f"multiscale for {site}"):
        return   # already extracted; skip the fetch

    lat, lon = coords(site)
    dlat = OUTER_KM / 111.32
    dlon = OUTER_KM / (111.32 * np.cos(np.radians(lat)))
    bbox = [lon - dlon, lat - dlat, lon + dlon, lat + dlat]
    print(f"{site}  ({lat:.4f}, {lon:.4f})  {start}..{end}  radii={RADII} m")

    y0, y1 = int(start[:4]), int(end[:4])
    frames = []
    for coll in COLLECTIONS:
        got = []
        for yr in range(y0, y1 + 1):
            for attempt in range(1, MAX_TRIES + 1):
                try:
                    da = fetch(coll, bbox, f"{yr}-01-01", f"{yr}-12-31")
                    if da is not None:
                        got.append(radial_means(da, coll, lat, lon))
                    break
                except Exception as e:
                    if attempt == MAX_TRIES:
                        print(f"  {coll} {yr}: LOST ({type(e).__name__})")
                    else:
                        time.sleep(5 * attempt)
        if got:
            frames.append(pd.concat(got).sort_index())
            print(f"  {coll}: {sum(len(g) for g in got)} acquisitions")

    if not frames:
        raise SystemExit(f"{site}: nothing retrieved")

    df = pd.concat(frames, axis=1).sort_index()
    df = add_indices(df)

    lst = df.get(f"LST_K_r{RADII[-1]}")
    if lst is None or lst.notna().sum() == 0:
        raise SystemExit(f"{site}: no usable LST -- refusing to write")

    os.makedirs(OUT, exist_ok=True)
    dest = os.path.join(OUT, f"{site}_multiscale.parquet")
    df.to_parquet(dest)
    print(f"wrote {dest}  ({len(df)} acq, {len(df.columns)} features)")

    print("\n  how much does the answer depend on the radius?")
    print(f"  {'radius':>7} {'LST med(K)':>11} {'NDVI med':>9} {'MNDWI med':>10}")
    for r in RADII:
        l = df.get(f"LST_K_r{r}")
        n = df.get(f"NDVI_r{r}")
        m = df.get(f"MNDWI_r{r}")
        print(f"  {r:>5} m {l.median() if l is not None else float('nan'):>11.1f}"
              f" {n.median() if n is not None else float('nan'):>9.2f}"
              f" {m.median() if m is not None else float('nan'):>10.2f}")


if __name__ == "__main__":
    main()
