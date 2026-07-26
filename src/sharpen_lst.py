"""Sharpen Landsat LST to 10 m using Sentinel-2, via DMS (Gao et al. 2012).

Method: pyDMS DecisionTreeSharpener -- the same Data Mining Sharpener algorithm
that OpenET's sharpener implements. (OpenET's own package runs on Google Earth
Engine; we are on Planetary Computer + HPC, so we run the algorithm locally.)

WHAT SHARPENING DOES AND DOES NOT DO
  Landsat TIRS senses thermal at 100 m and delivers it resampled to 30 m. DMS
  learns the relationship between coarse LST and fine-resolution predictors
  (here: Sentinel-2 reflectance at 10 m) with regression trees, applies it at
  10 m, then adds back the residual so the sharpened field RE-AGGREGATES to the
  original coarse LST (energy is conserved, not invented).

  It therefore REDISTRIBUTES a 100 m thermal signal according to 10 m optical
  structure. It does not create thermal information that was never measured.

  The payoff is real where the coarse pixel is MIXED -- a marsh/open-water
  boundary, which is exactly the coastal problem. It is near-useless over
  homogeneous terrain (US-Elm), and most valuable at heterogeneous ones
  (US-Esm, US-EvM), which the multiscale extraction already flagged.

  CAUTION for the downstream ML: a 10 m sharpened LST is partly a FUNCTION of
  Sentinel-2 reflectance. If it improves ET prediction, you cannot cleanly claim
  "sharpening helped" versus "S2 predicts ET" without an ablation that includes
  raw S2 bands as a baseline. Build that ablation in.

Usage:  python sharpen_lst.py SITE_ID [START] [END]
Run via Slurm, never on the login node.
"""

import os
import sys
import glob
import shutil
import tempfile

os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "6")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
import pyproj
import planetary_computer as pc
import pystac_client
import stackstac

from pyDMS.pyDMS import DecisionTreeSharpener

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
FULL = "/anvil/scratch/x-jwang120/coastal-et/data/processed/us_sites_coastal_distance.csv"
OUT = "/anvil/scratch/x-jwang120/coastal-et/data/interim/sharpened"
TIF = "/anvil/scratch/x-jwang120/coastal-et/data/interim/sharpened_tif"

BOX_KM = 2.0            # half-width of the sharpening scene
S2_PAIR_DAYS = 3        # max |Landsat date - Sentinel-2 date|
FINE_RES = 10           # m, the target
COARSE_RES = 30         # m, the Landsat grid we sharpen FROM (100 m information)
RADII = [100, 250, 500, 1000]

S2_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]


def coords(site):
    r = pd.read_csv(FULL).query("SITE_ID == @site")
    if r.empty:
        raise SystemExit(f"{site}: no coordinates")
    return float(r.iloc[0].LAT), float(r.iloc[0].LON)


def utm_epsg(lat, lon):
    zone = int((lon + 180) // 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def write_tif(path, arr, transform, crs, nodata=np.nan):
    arr = np.atleast_3d(arr.transpose() if arr.ndim == 2 else arr)
    if arr.ndim == 2:
        arr = arr[None]
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1],
                       width=arr.shape[2], count=arr.shape[0], dtype="float32",
                       crs=crs, transform=transform, nodata=nodata) as d:
        d.write(arr.astype("float32"))
    return path


def stack_to_array(items, assets, bbox, epsg, res):
    da = stackstac.stack(items, assets=assets, bounds_latlon=bbox, epsg=epsg,
                         resolution=res, chunksize=1024)
    da = da.isel(time=0) if da.sizes.get("time", 1) == 1 else da.median("time")
    return da.compute()


def main():
    site = sys.argv[1]
    start = sys.argv[2] if len(sys.argv) > 2 else "2022-01-01"
    end = sys.argv[3] if len(sys.argv) > 3 else "2023-12-31"

    lat, lon = coords(site)
    epsg = utm_epsg(lat, lon)
    dlat = BOX_KM / 111.32
    dlon = BOX_KM / (111.32 * np.cos(np.radians(lat)))
    bbox = [lon - dlon, lat - dlat, lon + dlon, lat + dlat]
    print(f"{site}  ({lat:.4f},{lon:.4f})  UTM EPSG:{epsg}  {start}..{end}")

    cat = pystac_client.Client.open(STAC, modifier=pc.sign_inplace)
    ls = list(cat.search(collections=["landsat-c2-l2"], bbox=bbox,
                         datetime=f"{start}/{end}",
                         query={"eo:cloud_cover": {"lt": 60}}).items())
    s2 = list(cat.search(collections=["sentinel-2-l2a"], bbox=bbox,
                         datetime=f"{start}/{end}",
                         query={"eo:cloud_cover": {"lt": 60}}).items())
    # Landsat 7/5 name their thermal band "lwir", not "lwir11"; keep only scenes
    # that actually carry every asset we ask stackstac for, or the read KeyErrors.
    ls = [i for i in ls if {"lwir11", "qa_pixel"} <= set(i.assets)]
    s2 = [i for i in s2 if set(S2_BANDS) <= set(i.assets)]

    # one date can return several items (adjacent path/rows). Keep the least
    # cloudy per date, or we sharpen the same day repeatedly.
    def dedupe(items):
        best = {}
        for i in items:
            d = pd.Timestamp(i.datetime).tz_convert("UTC").normalize()
            cc = i.properties.get("eo:cloud_cover", 100)
            if d not in best or cc < best[d].properties.get("eo:cloud_cover", 100):
                best[d] = i
        return [best[d] for d in sorted(best)]

    ls, s2 = dedupe(ls), dedupe(s2)
    print(f"  usable Landsat dates: {len(ls)}   Sentinel-2 dates: {len(s2)}")
    if not ls or not s2:
        raise SystemExit("need both sensors")

    s2_dates = pd.to_datetime([i.datetime for i in s2], utc=True)

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(TIF, exist_ok=True)
    tx, ty = pyproj.Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)

    recs = []
    for it in ls:
        ldate = pd.Timestamp(it.datetime).tz_convert("UTC")
        # pair with the nearest Sentinel-2 scene in time
        gap = np.abs((s2_dates - ldate).total_seconds()) / 86400
        j = int(np.argmin(gap))
        if gap[j] > S2_PAIR_DAYS:
            continue

        tmp = tempfile.mkdtemp(dir=TIF)
        try:
            # ---- coarse: Landsat LST on the 30 m grid ----
            lst = stack_to_array([it], ["lwir11", "qa_pixel"], bbox, epsg, COARSE_RES)
            qa = lst.sel(band="qa_pixel").values.astype("uint16")
            t = lst.sel(band="lwir11").values.astype("float32")  # kelvin already
            t[(qa & 0b111110) > 0] = np.nan                      # cloud/shadow
            if np.isfinite(t).mean() < 0.5:
                continue                                          # too cloudy

            # ---- fine: Sentinel-2 reflectance on the 10 m grid ----
            hi = stack_to_array([s2[j]], S2_BANDS, bbox, epsg, FINE_RES)
            h = hi.values.astype("float32")
            if not np.isfinite(h).any():
                continue

            def tf(a, res):
                x0 = float(a.x.min()) - res / 2
                y1 = float(a.y.max()) + res / 2
                return from_origin(x0, y1, res, res)

            lo_p = write_tif(os.path.join(tmp, "lst_low.tif"), t[None],
                             tf(lst, COARSE_RES), f"EPSG:{epsg}")
            hi_p = write_tif(os.path.join(tmp, "s2_high.tif"), h,
                             tf(hi, FINE_RES), f"EPSG:{epsg}")

            # ---- DMS ----
            sharp = DecisionTreeSharpener(
                highResFiles=[hi_p], lowResFiles=[lo_p],
                lowResQualityFiles=[], lowResGoodQualityFlags=[],
                cvHomogeneityThreshold=0.0,   # let it pick adaptively
                movingWindowSize=0,           # global model; scene is small
                disaggregatingTemperature=True,   # LST, not a reflectance
                perLeafLinearRegression=True,
                linearRegressionExtrapolationRatio=0.25,
            )
            sharp.trainSharpener()
            ds = sharp.applySharpener(highResFilename=hi_p, lowResFilename=lo_p)
            resid, corrected = sharp.residualAnalysis(ds, lo_p, None,
                                                      doCorrection=True)
            fine = corrected.GetRasterBand(1).ReadAsArray().astype("float32")
            gt = corrected.GetGeoTransform()

            # sanity: sharpening must conserve energy, not invent it
            v = fine[np.isfinite(fine)]
            if not len(v) or not (240 < np.nanmedian(v) < 340):
                print(f"  {ldate:%Y-%m-%d}: sharpened LST implausible, skipping")
                continue

            # ---- tower-centred means at each radius, on the 10 m grid ----
            ny, nx = fine.shape
            xs = gt[0] + (np.arange(nx) + 0.5) * gt[1]
            ys = gt[3] + (np.arange(ny) + 0.5) * gt[5]
            X, Y = np.meshgrid(xs, ys)
            dist = np.sqrt((X - tx) ** 2 + (Y - ty) ** 2)

            rec = {"date": ldate.normalize(),
                   "s2_gap_days": round(float(gap[j]), 1),
                   "LST30_mean": float(np.nanmean(t)),
                   "LST10_mean": float(np.nanmean(fine)),
                   "LST10_std": float(np.nanstd(fine))}
            for r in RADII:
                m = dist <= r
                rec[f"LST10_r{r}"] = float(np.nanmean(fine[m])) if m.any() else np.nan
                # the coarse value at the same radius, for a like-for-like compare
                Xc, Yc = np.meshgrid(lst.x.values, lst.y.values)
                mc = np.sqrt((Xc - tx) ** 2 + (Yc - ty) ** 2) <= r
                rec[f"LST30_r{r}"] = float(np.nanmean(t[mc])) if mc.any() else np.nan
            recs.append(rec)

            out_tif = os.path.join(TIF, f"{site}_LST10m_{ldate:%Y%m%d}.tif")
            write_tif(out_tif, fine[None],
                      from_origin(gt[0], gt[3], gt[1], -gt[5]), f"EPSG:{epsg}")
            print(f"  {ldate:%Y-%m-%d}: sharpened 30m->10m  "
                  f"(S2 gap {gap[j]:.0f}d)  LST {np.nanmean(fine):.1f} K  "
                  f"sub-pixel sd {np.nanstd(fine):.2f} K", flush=True)
        except Exception as e:
            print(f"  {ldate:%Y-%m-%d}: FAILED {type(e).__name__}: {str(e)[:60]}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    if not recs:
        raise SystemExit(f"{site}: no dates sharpened")

    df = pd.DataFrame(recs).set_index("date").sort_index()
    dest = os.path.join(OUT, f"{site}_lst10m.parquet")
    df.to_parquet(dest)
    print(f"\nwrote {dest}  ({len(df)} sharpened dates)")
    print(f"  mean |LST10 - LST30| at 100 m radius: "
          f"{(df['LST10_r100'] - df['LST30_r100']).abs().mean():.2f} K")
    print("  ^ how much the sharpening actually MOVES the tower-scale value.")
    print("    Near zero => homogeneous pixel, sharpening bought you nothing.")


if __name__ == "__main__":
    main()
