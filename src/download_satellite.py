"""Extract satellite predictor chips around each coastal flux tower.

Pulls Landsat C2-L2 (30 m SR + surface temperature), Sentinel-2 L2A (10-20 m
optical) and Sentinel-1 RTC (VV/VH backscatter -> inundation proxy) from the
Microsoft Planetary Computer STAC, clipped to a chip centred on each tower and
sized to comfortably contain the EC footprint.

Sentinel-1 is the coastal-specific input: optical and thermal cannot see water
standing *under* a marsh canopy, so without SAR the model is blind to the single
variable that most controls marsh energy partitioning.

Usage:  python download_satellite.py SITE_ID [START] [END]
Run via Slurm (one array task per site), never on the login node.
"""

import os
import sys
import csv
import time

# GDAL/vsicurl tuning must be set BEFORE rasterio is imported. Planetary Computer
# range-reads fail intermittently under concurrency; without retries a single bad
# read raises and (previously) discarded an entire site's Landsat record.
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "6")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "3")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "120")
# NB: do NOT set CPL_VSIL_CURL_ALLOWED_EXTENSIONS. It looks like a harmless
# optimisation, but the three sensors do not agree on a suffix --
#   Landsat     ..._SR_B4.TIF
#   Sentinel-2  ..._B04_10m.tif
#   Sentinel-1  iw-vv.rtc.tiff      <-- four letters
# An allowlist of ".tif,.TIF" silently blocks every Sentinel-1 read, which is the
# one sensor that can see water under a marsh canopy. Failing open is safer here.

import numpy as np
import pandas as pd
import dask
import planetary_computer as pc
import pystac_client
import stackstac

# Chip reads are network-latency-bound, not CPU-bound: one site is ~7,500 HTTP
# range reads. The dask default (one thread per core) leaves the network almost
# idle, so we deliberately oversubscribe.
dask.config.set(scheduler="threads", num_workers=32)

MAX_TRIES = 3

# A scene is only usable if enough of the chip SURVIVES the cloud mask. Masking
# alone is not enough: a 95%-clouded scene still leaves a few pixels around the
# cloud edges, and their mean is noise, not a measurement. 30% of Sentinel-2
# scenes at US-Skr had under 25% valid pixels and were being silently averaged.
MIN_VALID_FRAC = 0.50

SITES_CSV = "/anvil/scratch/x-jwang120/coastal-et/config/coastal_sites.csv"
OUT = "/anvil/scratch/x-jwang120/coastal-et/data/interim/satellite"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"

CHIP_KM = 3.0  # half-width; 6x6 km box comfortably contains any EC footprint
EPSG = 3857    # metric CRS for chipping; fine at chip scale

COLLECTIONS = {
    # Landsat is non-negotiable: it is the ONLY source of thermal (LST) here.
    # Sentinel-2 has no thermal band, so it cannot anchor an energy-balance model.
    "landsat": dict(
        name="landsat-c2-l2",
        assets=["red", "nir08", "green", "swir16", "lwir11", "qa_pixel"],
        res=30,
    ),
    "sentinel2": dict(
        name="sentinel-2-l2a",
        assets=["B03", "B04", "B08", "B11", "SCL"],
        res=20,
    ),
}

# Sentinel-1 (SAR) is opt-in: SAT_INCLUDE_S1=1. It is the only sensor that sees
# standing water UNDER a canopy -- real value at a tidally flooded mangrove like
# US-Skr -- but it is a value-add, not a requirement, and it roughly doubles the
# download. The default design here is Sentinel-2 + Landsat + ERA5.
if os.environ.get("SAT_INCLUDE_S1") == "1":
    COLLECTIONS["sentinel1"] = dict(
        name="sentinel-1-rtc", assets=["vv", "vh"], res=20,
    )


def site_row(site_id):
    with open(SITES_CSV) as f:
        for r in csv.DictReader(f):
            if r["SITE_ID"] == site_id:
                return r
    raise SystemExit(f"site {site_id} not in {SITES_CSV}")


def bbox_around(lat, lon, half_km=CHIP_KM):
    dlat = half_km / 111.32
    dlon = half_km / (111.32 * np.cos(np.radians(lat)))
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]


def fetch(collection, bbox, start, end, quiet=False):
    cfg = COLLECTIONS[collection]
    # re-open (and therefore re-sign) each call: PC's SAS tokens expire, and a
    # long multi-year job outlives a token minted at the start
    cat = pystac_client.Client.open(STAC, modifier=pc.sign_inplace)
    search = cat.search(
        collections=[cfg["name"]],
        bbox=bbox,
        datetime=f"{start}/{end}",
    )
    items = list(search.items())
    if not items:
        if not quiet:
            print(f"  {collection}: no scenes")
        return None
    if not quiet:
        print(f"  {collection}: {len(items)} scenes")
    da = stackstac.stack(
        items,
        assets=cfg["assets"],
        bounds_latlon=bbox,
        epsg=EPSG,
        resolution=cfg["res"],
        chunksize=512,
    )
    return da


def apply_qa_mask(da, collection):
    """Drop cloud/shadow-contaminated pixels before any spatial averaging.

    Without this the chip mean folds cloud tops into LST -- a fully clouded
    scene averages to ~225 K, which is not a land surface temperature and would
    silently poison every thermal feature.
    """
    bands = list(da.band.values)

    if collection == "landsat":
        if "qa_pixel" not in bands:
            return da
        qa = da.sel(band="qa_pixel").fillna(1).astype("uint16")
        # C2 QA_PIXEL bits: 1 dilated cloud, 2 cirrus, 3 cloud, 4 shadow, 5 snow
        bad = (qa & 0b111110) > 0
        keep = [b for b in bands if b != "qa_pixel"]
        return da.sel(band=keep).where(~bad)

    if collection == "sentinel2":
        if "SCL" not in bands:
            return da
        scl = da.sel(band="SCL")
        # SCL 4 veg, 5 bare, 6 water, 7 unclassified; everything else is
        # cloud, shadow, cirrus, snow, saturated or nodata
        good = scl.isin([4, 5, 6, 7])
        keep = [b for b in bands if b != "SCL"]
        return da.sel(band=keep).where(good)

    return da  # Sentinel-1 SAR is unaffected by cloud -- that is the point of it


def try_window(coll, bbox, ys, ye, got):
    """Fetch one time window with retries; append to `got`. True if it succeeded."""
    for attempt in range(1, MAX_TRIES + 1):
        try:
            f = summarize(fetch(coll, bbox, ys, ye, quiet=True), coll)
            if not f.empty:
                got.append(f)
            return True
        except Exception:
            if attempt < MAX_TRIES:
                time.sleep(5 * attempt)  # back off; these are usually transient
    return False


def pixel_indices(da, collection):
    """Per-pixel NDVI/MNDWI with physical guards, computed BEFORE averaging.

    Averaging the bands and then taking the ratio is NOT the same as taking the
    ratio per pixel and then averaging -- the index is nonlinear. At US-TaS (a
    slough, so the chip is full of water) that difference put NDVI at 0.51
    instead of 0.23: more than double.

    Guards, both essential over water:
      * Landsat C2 surface reflectance uses offset=-0.2, so dark water goes
        NEGATIVE. 16% of NIR pixels at US-TaS are below zero.
      * a near-zero denominator makes the ratio explode -- 17% of raw per-pixel
        NDVI at US-TaS landed outside [-1, 1], some as high as 190.
    """
    import xarray as xr
    bands = list(da.band.values)

    def get(b):
        return da.sel(band=b) if b in bands else None

    pairs = {}
    if collection == "landsat":
        pairs["NDVI"] = (get("nir08"), get("red"))
        pairs["MNDWI"] = (get("green"), get("swir16"))
    elif collection == "sentinel2":
        pairs["NDVI_S2"] = (get("B08"), get("B04"))
        pairs["MNDWI_S2"] = (get("B03"), get("B11"))

    out = {}
    for name, (a, b) in pairs.items():
        if a is None or b is None:
            continue
        # reflectance must be physical; negatives are noise over dark water
        a = a.where(a > 0)
        b = b.where(b > 0)
        den = a + b
        idx = (a - b) / den.where(den > 0.02)     # guard the denominator
        out[name] = idx.where(abs(idx) <= 1.0)    # guard the range
    return out


def summarize(da, collection):
    """Reduce each chip to tower-centred means + spatial texture stats."""
    if da is None:
        return pd.DataFrame()
    da = apply_qa_mask(da, collection)

    # drop acquisitions where too little of the chip survived the cloud mask
    ref = {"landsat": "lwir11", "sentinel2": "B08", "sentinel1": "vv"}.get(collection)
    valid_frac = None
    if ref is not None and ref in list(da.band.values):
        valid_frac = da.sel(band=ref).notnull().mean(dim=("y", "x"))

    idx = pixel_indices(da, collection)
    # chip mean and std per band per acquisition; std is the sub-pixel
    # heterogeneity signal that matters for mixed marsh/water pixels.
    # Compute BOTH in one dask graph: called separately they build two graphs and
    # every chip gets downloaded twice, doubling the (network-bound) runtime.
    tasks = [da.mean(dim=("y", "x"), skipna=True),
             da.std(dim=("y", "x"), skipna=True)]
    have_vf = valid_frac is not None
    if have_vf:
        tasks.append(valid_frac)
    keys = list(idx)
    tasks += [idx[k].mean(dim=("y", "x"), skipna=True) for k in keys]
    tasks += [idx[k].std(dim=("y", "x"), skipna=True) for k in keys]
    res = dask.compute(*tasks)
    mean, std = res[0], res[1]
    off = 2
    vf = None
    if have_vf:
        vf = res[2]
        off = 3
    n = len(keys)
    idx_mean = {k: res[off + i] for i, k in enumerate(keys)}
    idx_std = {k: res[off + n + i] for i, k in enumerate(keys)}

    df = mean.to_pandas()
    df.columns = [f"{collection}_{b}_mean" for b in df.columns]
    ds = std.to_pandas()
    ds.columns = [f"{collection}_{b}_std" for b in ds.columns]
    out = df.join(ds)
    for k in keys:
        out[k] = idx_mean[k].to_pandas()
        out[f"{k}_sd"] = idx_std[k].to_pandas()   # within-chip heterogeneity
    out.index.name = "time"

    # One overpass can yield several STAC items -- adjacent tiles/swaths whose
    # footprints both touch the chip -- so timestamps repeat. Collapse them, or
    # the index is non-unique and the cross-sensor join below cannot align.
    out = out.groupby(level=0).mean().sort_index()

    # a scene that was entirely cloud is now all-NaN: it carries no information
    out = out.dropna(how="all")

    if vf is not None:
        f = vf.to_pandas()
        f = f.groupby(level=0).max()
        out["valid_frac"] = f.reindex(out.index)
        # keep the fraction as a column (it is a useful quality weight) but drop
        # anything below the floor: those means are cloud-edge noise
        out = out[out["valid_frac"].fillna(0) >= MIN_VALID_FRAC]
    return out


def add_indices(df):
    def col(c):
        return df[c] if c in df else None

    # NDVI / MNDWI are now computed PER PIXEL in pixel_indices() and are already
    # columns here. Recomputing them from the chip-mean bands would reintroduce
    # the ratio-of-means bias.
    if "landsat_lwir11_mean" in df:
        # stackstac already applies the scale/offset in the item's raster:bands
        # metadata (scale 0.00341802, offset 149.0), so lwir11 arrives in kelvin.
        # Applying it a second time yields ~150 K, which is not a temperature.
        df["LST_K"] = df["landsat_lwir11_mean"]
        lst = df["LST_K"].dropna()
        if len(lst) and not (240 < lst.median() < 330):
            raise ValueError(
                f"LST median {lst.median():.1f} K is outside any plausible land "
                "surface range -- check the scale/offset handling."
            )
    if "sentinel1_vv_mean" in df and "sentinel1_vh_mean" in df:
        df["S1_VV_dB"] = 10 * np.log10(df["sentinel1_vv_mean"].clip(lower=1e-6))
        df["S1_VH_dB"] = 10 * np.log10(df["sentinel1_vh_mean"].clip(lower=1e-6))
        df["S1_RATIO"] = df["S1_VV_dB"] - df["S1_VH_dB"]
    return df


import io_utils  # idempotent skip/overwrite helper (lives in the same src/ dir)


def main():
    argv = io_utils.clean_argv()
    if len(argv) < 2:
        raise SystemExit(__doc__)
    site_id = argv[1]
    start = argv[2] if len(argv) > 2 else "2015-01-01"
    end = argv[3] if len(argv) > 3 else "2025-12-31"

    os.makedirs(OUT, exist_ok=True)
    dest = os.path.join(OUT, f"{site_id}_satellite.parquet")
    if not io_utils.should_write(dest, label=f"satellite for {site_id}"):
        return   # already downloaded; skip the whole fetch

    row = site_row(site_id)
    # coastal_sites.csv is the curated stratification; coordinates live in the
    # inventory written by coastal_sites.py
    full = pd.read_csv(
        "/anvil/scratch/x-jwang120/coastal-et/data/processed/us_sites_coastal_distance.csv"
    )
    r = full[full.SITE_ID == site_id].iloc[0]
    lat, lon = float(r.LAT), float(r.LON)

    print(f"{site_id}  ({lat:.4f}, {lon:.4f})  {start}..{end}  stratum={row['STRATUM']}")
    bbox = bbox_around(lat, lon)

    # Fetch year by year. The Planetary Computer throws transient read errors, and
    # a decade-long stack means one bad scene would otherwise cost us every year.
    # Isolating per year bounds the blast radius to that year, and we retry it.
    y0, y1 = int(start[:4]), int(end[:4])
    frames = []
    for coll in COLLECTIONS:
        got, lost = [], []
        t_coll = time.time()
        for yr in range(y0, y1 + 1):
            t0 = time.time()
            ok = try_window(coll, bbox, f"{yr}-01-01", f"{yr}-12-31", got)
            print(f"    {coll} {yr}: {'ok ' if ok else 'RETRY'} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if ok:
                continue
            # A persistently bad year usually means ONE unreadable scene, not a
            # bad year. Fall back to months so we lose that month, not the year.
            bad_months = []
            for m in range(1, 13):
                last = pd.Period(f"{yr}-{m:02d}").days_in_month
                if not try_window(coll, bbox, f"{yr}-{m:02d}-01",
                                  f"{yr}-{m:02d}-{last}", got):
                    bad_months.append(m)
            if bad_months:
                lost.append(f"{yr}m{bad_months}")
        if got:
            frames.append(pd.concat(got).sort_index())
        n = sum(len(f) for f in got)
        msg = (f"  {coll}: {n} acquisitions over {y1 - y0 + 1 - len(lost)} yrs"
               f"  [{time.time() - t_coll:.0f}s]")
        if lost:
            msg += f"  [LOST YEARS: {lost}]"   # never let this pass silently
        print(msg)

    frames = [f for f in frames if not f.empty]
    if not frames:
        raise SystemExit(f"{site_id}: no satellite data retrieved")

    df = pd.concat(frames, axis=1).sort_index()
    df = add_indices(df)

    # LST is the anchor of every energy-balance model here. A site with none is
    # useless for the thermal tier, so fail loudly rather than write a file that
    # looks fine until someone tries to train on it.
    n_lst = int(df["LST_K"].notna().sum()) if "LST_K" in df else 0
    n_sar = int(df["S1_VV_dB"].notna().sum()) if "S1_VV_dB" in df else 0

    # Guard EVERY sensor that the model cannot do without, not just the thermal
    # one. A silent loss of Sentinel-1 already happened once (a GDAL extension
    # allowlist blocked .tiff) and produced a full, plausible-looking dataset
    # with no SAR in it at all.
    missing = []
    if n_lst == 0:
        missing.append("Landsat LST (thermal anchor)")
    if "sentinel1" in COLLECTIONS and n_sar == 0:
        missing.append("Sentinel-1 SAR (sub-canopy inundation)")
    if missing:
        raise SystemExit(
            f"{site_id}: ZERO usable data for: {'; '.join(missing)}. "
            "Refusing to write a parquet that is missing a required sensor."
        )

    os.makedirs(OUT, exist_ok=True)
    dest = os.path.join(OUT, f"{site_id}_satellite.parquet")
    df.to_parquet(dest)
    print(f"wrote {dest}  ({len(df)} acquisitions, {len(df.columns)} features, "
          f"{n_lst} usable LST, {n_sar} SAR)")


if __name__ == "__main__":
    main()
