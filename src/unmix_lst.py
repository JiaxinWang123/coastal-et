"""Does sharpening actually buy us anything? Test it, do not assume it.

The DMS residual correction forces the sharpened field to re-aggregate to the
coarse LST. So at any radius >= the coarse pixel, mean(LST10) == mean(LST30) BY
CONSTRUCTION -- which is exactly what we measured (|LST10-LST30| ~ 0.01 K).

Therefore: a model fed footprint-MEAN LST gains nothing from sharpening. Ever.

What sharpening can give you is the WITHIN-pixel distribution, i.e. the ability to
compute the temperature of the MARSH alone, separate from the open water mixed
into the same coarse pixel. Coastal towers sit on exactly such boundaries, and
water (high thermal inertia, evaporating at potential rate) has a very different
temperature from vegetated marsh.

This script quantifies that gain:

    LST_all    what an unsharpened model sees (footprint mean, all cover)
    LST_veg    temperature of the vegetated fraction only  <- what TSEB wants
    LST_water  temperature of the open-water fraction
    dT         LST_veg - LST_all   = what unmixing changes

If dT is ~0, sharpening is cosmetic here and should be dropped. If dT is large,
it is the whole point.

Run via Slurm, never on the login node.
"""

import os
import glob

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

import numpy as np
import pandas as pd
import rasterio
import pyproj
import planetary_computer as pc
import pystac_client
import stackstac

TIF = "/anvil/scratch/x-jwang120/coastal-et/data/interim/sharpened_tif"
OUT = "/anvil/scratch/x-jwang120/coastal-et/data/processed"
FULL = "/anvil/scratch/x-jwang120/coastal-et/data/processed/us_sites_coastal_distance.csv"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"

RADIUS = 250          # m; ~the marsh footprint scale
MNDWI_WATER = 0.0     # MNDWI > 0 => open water
MAX_DATES = 8         # per site; this is a diagnostic, not a product


def coords(site):
    r = pd.read_csv(FULL).query("SITE_ID == @site")
    return float(r.iloc[0].LAT), float(r.iloc[0].LON)


def s2_indices(lat, lon, date, epsg, shape, transform):
    """NDVI + MNDWI at 10 m on the sharpened grid, for the nearest S2 scene."""
    d = 2.0 / 111.32
    dl = 2.0 / (111.32 * np.cos(np.radians(lat)))
    bbox = [lon - dl, lat - d, lon + dl, lat + d]
    cat = pystac_client.Client.open(STAC, modifier=pc.sign_inplace)
    lo = (date - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    hi = (date + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    items = [i for i in cat.search(collections=["sentinel-2-l2a"], bbox=bbox,
                                   datetime=f"{lo}/{hi}").items()
             if {"B03", "B04", "B08", "B11"} <= set(i.assets)]
    if not items:
        return None, None
    items.sort(key=lambda i: abs(pd.Timestamp(i.datetime).tz_convert("UTC") - date))
    da = stackstac.stack([items[0]], assets=["B03", "B04", "B08", "B11"],
                         bounds_latlon=bbox, epsg=epsg, resolution=10).isel(time=0)
    a = da.compute()
    g = a.sel(band="B03").values.astype("float32")
    r_ = a.sel(band="B04").values.astype("float32")
    n = a.sel(band="B08").values.astype("float32")
    s = a.sel(band="B11").values.astype("float32")
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (n - r_) / (n + r_)
        mndwi = (g - s) / (g + s)
    # crop/pad to the sharpened grid
    H, W = shape
    ndvi = ndvi[:H, :W]
    mndwi = mndwi[:H, :W]
    if ndvi.shape != shape:
        return None, None
    return ndvi, mndwi


def main():
    rows = []
    for site in ["US-Skr", "US-EvM", "US-Esm", "US-Elm"]:
        lat, lon = coords(site)
        tifs = sorted(glob.glob(os.path.join(TIF, f"{site}_LST10m_*.tif")))[:MAX_DATES]
        if not tifs:
            continue
        print(f"{site}: {len(tifs)} sharpened dates")
        for p in tifs:
            date = pd.Timestamp(os.path.basename(p).split("_")[-1][:8], tz="UTC")
            with rasterio.open(p) as src:
                lst = src.read(1).astype("float32")
                epsg = src.crs.to_epsg()
                tr = src.transform
                H, W = lst.shape
            ndvi, mndwi = s2_indices(lat, lon, date, epsg, (H, W), tr)
            if ndvi is None:
                continue

            tx, ty = pyproj.Transformer.from_crs(
                4326, epsg, always_xy=True).transform(lon, lat)
            xs = tr.c + (np.arange(W) + 0.5) * tr.a
            ys = tr.f + (np.arange(H) + 0.5) * tr.e
            X, Y = np.meshgrid(xs, ys)
            near = np.sqrt((X - tx) ** 2 + (Y - ty) ** 2) <= RADIUS

            water = near & (mndwi > MNDWI_WATER) & np.isfinite(lst)
            veg = near & (mndwi <= MNDWI_WATER) & np.isfinite(lst)
            allp = near & np.isfinite(lst)
            if allp.sum() < 50 or veg.sum() < 10:
                continue

            t_all = float(np.nanmean(lst[allp]))
            t_veg = float(np.nanmean(lst[veg]))
            t_wat = float(np.nanmean(lst[water])) if water.sum() >= 10 else np.nan
            rows.append(dict(
                site=site, date=date.date(),
                water_frac=round(float(water.sum() / allp.sum()), 3),
                LST_all=round(t_all, 2), LST_veg=round(t_veg, 2),
                LST_water=round(t_wat, 2) if np.isfinite(t_wat) else np.nan,
                dT=round(t_veg - t_all, 2)))

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no dates evaluated")
    df.to_csv(os.path.join(OUT, "sharpening_unmixing_test.csv"), index=False)
    print("\n" + df.to_string(index=False))

    print("\n=== WHAT DOES SHARPENING ACTUALLY BUY? ===")
    print(f"(radius {RADIUS} m; dT = LST_veg - LST_all, in kelvin)")
    g = df.groupby("site").agg(
        n=("dT", "size"), water_frac=("water_frac", "mean"),
        dT_mean=("dT", "mean"), dT_max=("dT", lambda s: s.abs().max()),
        veg_minus_water=("LST_water", lambda s: np.nan))
    for s in g.index:
        d = df[df.site == s]
        vw = (d.LST_veg - d.LST_water).mean()
        g.loc[s, "veg_minus_water"] = round(vw, 2) if np.isfinite(vw) else np.nan
    print(g.round(3).to_string())
    print("\ninterpretation:")
    print("  water_frac ~0  -> the coarse pixel is NOT mixed; sharpening is cosmetic.")
    print("  |dT| large     -> unmixing materially changes the temperature the model")
    print("                    should be using, and sharpening is doing real work.")


if __name__ == "__main__":
    main()
