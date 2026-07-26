"""Extract satellite predictors over a small window centred on the flux tower.

The previous extraction used a 6 x 6 km chip -- vastly larger than any EC
footprint, and at the marsh sites it averaged in ~325x too much area. This one
represents the "tower point" by averaging the 30 m pixels over a WINDOW_M box
(default 90 m = 3x3 Landsat pixels).

A caveat worth carrying: the true EC footprint scales as ~100 x (z_m - d).

    marsh towers  (z_m ~4 m,  h_c ~1 m)   ->  ~300 m
    US-Skr        (z_m ~27 m, h_c ~20 m)  ->  ~1.4 km

So a 90 m window is SMALLER than what any of these towers actually sees, and at
US-Skr it is ~15x too small in linear extent. It is a clean, reproducible point
definition -- not a footprint model. We also emit the same statistics at 250 m
and 500 m so the scale sensitivity can be tested rather than assumed.

Sensors: Landsat C2-L2 (optical + THERMAL) and Sentinel-2 L2A (optical).
Cloud handling: QA_PIXEL / SCL masking, PLUS a minimum valid-pixel fraction --
a 95%-clouded scene still leaves cloud-edge pixels whose mean is noise.

Usage:  python extract_tower_point.py SITE_ID [START] [END]
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
import pyproj
import planetary_computer as pc
import pystac_client
import stackstac

dask.config.set(scheduler="threads", num_workers=32)

R = "/anvil/scratch/x-jwang120/coastal-et"
OUT = f"{R}/data/interim/tower_point"
FULL = f"{R}/data/processed/us_sites_coastal_distance.csv"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"

WINDOWS_M = [90, 250, 500]     # 90 m = 3x3 Landsat pixels = the requested "point"
PRIMARY = 90
DOWNLOAD_KM = 0.8              # pull once at the largest window + margin
MIN_VALID_FRAC = 0.50
MAX_TRIES = 3

COLLECTIONS = {
    "landsat": dict(name="landsat-c2-l2", res=30,
                    assets=["red", "nir08", "green", "swir16", "lwir11", "qa_pixel"]),
    "sentinel2": dict(name="sentinel-2-l2a", res=10,
                      assets=["B02", "B03", "B04", "B08", "B11", "B12", "SCL"]),
}


def coords(site):
    r = pd.read_csv(FULL).query("SITE_ID == @site")
    if r.empty:
        raise SystemExit(f"{site}: no coordinates")
    return float(r.iloc[0].LAT), float(r.iloc[0].LON)


def qa_mask(da, coll):
    bands = list(da.band.values)
    if coll == "landsat" and "qa_pixel" in bands:
        qa = da.sel(band="qa_pixel").fillna(1).astype("uint16")
        bad = (qa & 0b111110) > 0          # dilated cloud, cirrus, cloud, shadow, snow
        keep = [b for b in bands if b != "qa_pixel"]
        return da.sel(band=keep).where(~bad)
    if coll == "sentinel2" and "SCL" in bands:
        scl = da.sel(band="SCL")
        good = scl.isin([4, 5, 6, 7])      # veg, bare, water, unclassified
        keep = [b for b in bands if b != "SCL"]
        return da.sel(band=keep).where(good)
    return da


def indices(da, coll):
    """Per-pixel indices with guards, BEFORE any spatial averaging.

    Landsat C2 SR carries offset=-0.2, so dark water goes negative and a naive
    ratio explodes. Guard the reflectance, the denominator and the range.
    """
    bands = list(da.band.values)

    def g(b):
        return da.sel(band=b) if b in bands else None

    pairs = ({"NDVI": (g("nir08"), g("red")), "MNDWI": (g("green"), g("swir16"))}
             if coll == "landsat" else
             {"NDVI_S2": (g("B08"), g("B04")), "MNDWI_S2": (g("B03"), g("B11")),
              "NDWI_S2": (g("B03"), g("B08"))})
    out = {}
    for k, (a, b) in pairs.items():
        if a is None or b is None:
            continue
        a, b = a.where(a > 0), b.where(b > 0)
        den = a + b
        idx = (a - b) / den.where(den > 0.02)
        out[k] = idx.where(abs(idx) <= 1.0)
    return out


def fetch(coll, bbox, y0, y1):
    cfg = COLLECTIONS[coll]
    cat = pystac_client.Client.open(STAC, modifier=pc.sign_inplace)
    items = [i for i in cat.search(collections=[cfg["name"]], bbox=bbox,
                                   datetime=f"{y0}/{y1}").items()
             if set(cfg["assets"]) <= set(i.assets)]
    if not items:
        return None
    return stackstac.stack(items, assets=cfg["assets"], bounds_latlon=bbox,
                           epsg=3857, resolution=cfg["res"], chunksize=512)


def window_stats(da, coll, lat, lon):
    da = qa_mask(da, coll)
    idx = indices(da, coll)

    # EPSG:3857 inflates distance by 1/cos(lat) -- correct it or a "90 m" window
    # is really 100 m at this latitude.
    tx, ty = pyproj.Transformer.from_crs(4326, 3857, always_xy=True).transform(lon, lat)
    X, Y = np.meshgrid(da.x.values, da.y.values)
    dist = np.maximum(np.abs(X - tx), np.abs(Y - ty)) * np.cos(np.radians(lat))

    import xarray as xr
    frames = []
    for w in WINDOWS_M:
        m = xr.DataArray(dist <= w / 2.0, dims=("y", "x"),
                         coords={"y": da.y, "x": da.x})
        sub = da.where(m)
        tasks, names = [], []

        # valid_frac must be normalised by the WINDOW, not the chip. Taking a
        # mean over (y, x) after .where(m) divides by every pixel in the 60x60
        # chip, so a full 3x3 window scored 9/3600 = 0.002 and the >=0.5 gate
        # threw away every scene.
        n_in_window = int(m.values.sum())
        ref = "lwir11" if coll == "landsat" else "B08"
        if ref in list(da.band.values) and n_in_window > 0:
            tasks.append(sub.sel(band=ref).notnull().sum(dim=("y", "x"))
                         / n_in_window)
            names.append(f"{coll}_valid_frac_w{w}")

        tasks.append(sub.mean(dim=("y", "x"), skipna=True))
        names.append("__bands__")
        for k, v in idx.items():
            tasks.append(v.where(m).mean(dim=("y", "x"), skipna=True))
            names.append(k)

        res = dask.compute(*tasks)
        cols = {}
        for nm, r_ in zip(names, res):
            if nm == "__bands__":
                b = r_.to_pandas()
                for c in b.columns:
                    cols[f"{coll}_{c}_w{w}"] = b[c]
            elif "valid_frac" in nm:
                cols[nm] = r_.to_pandas()          # already carries coll + window
            else:
                cols[f"{nm}_w{w}"] = r_.to_pandas()
        frames.append(pd.DataFrame(cols))

    out = pd.concat(frames, axis=1)
    out.index.name = "time"
    out = out.groupby(level=0).mean().sort_index().dropna(how="all")

    vf = f"{coll}_valid_frac_w{PRIMARY}"
    if vf in out:
        out = out[out[vf].fillna(0) >= MIN_VALID_FRAC]
    return out


import io_utils  # idempotent skip/overwrite helper (same src/ dir)


def main():
    argv = io_utils.clean_argv()
    site = argv[1]
    start = argv[2] if len(argv) > 2 else "2018-01-01"
    end = argv[3] if len(argv) > 3 else "2023-12-31"

    os.makedirs(OUT, exist_ok=True)
    p = f"{OUT}/{site}_towerpoint.parquet"
    if not io_utils.should_write(p, label=f"tower-point for {site}"):
        return   # already extracted; skip the fetch

    lat, lon = coords(site)
    d = DOWNLOAD_KM / 111.32
    dl = DOWNLOAD_KM / (111.32 * np.cos(np.radians(lat)))
    bbox = [lon - dl, lat - d, lon + dl, lat + d]
    print(f"{site} ({lat:.4f},{lon:.4f})  {start}..{end}  windows={WINDOWS_M} m")

    y0, y1 = int(start[:4]), int(end[:4])
    frames = []
    for coll in COLLECTIONS:
        got = []
        for yr in range(y0, y1 + 1):
            for a in range(1, MAX_TRIES + 1):
                try:
                    da = fetch(coll, bbox, f"{yr}-01-01", f"{yr}-12-31")
                    if da is not None:
                        got.append(window_stats(da, coll, lat, lon))
                    break
                except Exception as e:
                    if a == MAX_TRIES:
                        print(f"  {coll} {yr}: LOST ({type(e).__name__})")
                    else:
                        time.sleep(5 * a)
        if got:
            f = pd.concat(got).sort_index()
            frames.append(f)
            print(f"  {coll}: {len(f)} usable acquisitions")

    if not frames:
        raise SystemExit(f"{site}: nothing retrieved")
    df = pd.concat(frames, axis=1).sort_index()

    if "landsat_lwir11_w90" in df:
        df["LST_K"] = df["landsat_lwir11_w90"]     # stackstac already rescales
        lst = df["LST_K"].dropna()
        if len(lst) and not (240 < lst.median() < 330):
            raise ValueError(f"LST median {lst.median():.1f} K implausible")
        for w in WINDOWS_M:
            c = f"landsat_lwir11_w{w}"
            if c in df:
                df[f"LST_K_w{w}"] = df[c]

    n_lst = int(df["LST_K"].notna().sum()) if "LST_K" in df else 0
    if n_lst == 0:
        raise SystemExit(f"{site}: ZERO usable LST -- refusing to write")

    os.makedirs(OUT, exist_ok=True)
    p = f"{OUT}/{site}_towerpoint.parquet"
    df.to_parquet(p)
    print(f"wrote {p}  ({len(df)} acq, {len(df.columns)} cols, {n_lst} usable LST)")

    print("\n  scale sensitivity (median over the record):")
    print(f"  {'window':>7} {'LST (K)':>9} {'NDVI':>7} {'MNDWI':>8}")
    for w in WINDOWS_M:
        l = df.get(f"LST_K_w{w}")
        n = df.get(f"NDVI_w{w}")
        m = df.get(f"MNDWI_w{w}")
        f2 = lambda x: x.median() if x is not None else np.nan
        print(f"  {w:>5} m {f2(l):>9.1f} {f2(n):>7.2f} {f2(m):>8.2f}")


if __name__ == "__main__":
    main()
