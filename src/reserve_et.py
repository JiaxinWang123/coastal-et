"""Portable reserve-ET prediction pipeline (root derived from __file__).

Predict daily ET inside a reserve polygon with our validated 13-site ExtraTrees model:
clear Landsat scene -> 7 satellite indices per 30 m pixel -> gridMET point met ->
ExtraTrees -> clip to polygon, with open water masked out.

Used by both src/map_reserves.py (batch) and notebooks/04_spatial_prediction.ipynb.
No personal paths: ROOT is the project folder that contains this src/ directory.
"""
import os
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "6")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import dask
import planetary_computer as pc
import pystac_client
import stackstac
import rioxarray  # noqa: F401  (.rio accessor)
from rasterio.features import geometry_mask
from affine import Affine
from sklearn.ensemble import ExtraTreesRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # <proj>/src -> <proj>
# reserve boundaries live beside the project (team2/shp_predict); fall back to the
# shared location if we are running from a copy that doesn't sit next to them.
# reserve boundaries: look inside the project first (self-contained bundle), then
# beside it (team2 layout), then the shared Anvil location.
SHP_DIR = next((p for p in [
    os.path.join(ROOT, "shp_predict"),
    os.path.join(os.path.dirname(ROOT), "shp_predict"),
    "/anvil/projects/x-ees260113/team2/shp_predict",
] if os.path.isdir(p)), os.path.join(ROOT, "shp_predict"))
PROC = os.path.join(ROOT, "data", "processed")

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
ASSETS = ["red", "nir08", "green", "swir16", "lwir11", "qa_pixel"]
FEATS = ["LAI", "EVI2", "SAVI", "NDVI", "NDWI", "MNDWI", "LST_K",
         "TA_ERA", "VPD_ERA", "SW_IN_ERA", "WS_ERA", "ETo_mm", "DOY_sin", "DOY_cos"]
GM = "http://thredds.northwestknowledge.net:8080/thredds/dodsC/agg_met_{v}_1979_CurrentYear_CONUS.nc"
GVARS = {"tmmx": "daily_maximum_temperature", "tmmn": "daily_minimum_temperature",
         "rmax": "daily_maximum_relative_humidity", "rmin": "daily_minimum_relative_humidity",
         "vs": "daily_mean_wind_speed", "srad": "daily_mean_shortwave_radiation_at_surface",
         "pet": "daily_mean_reference_evapotranspiration_grass"}


def es_kpa(T):
    return 0.6108 * np.exp(17.27 * T / (T + 237.3))


def train_best():
    d = pd.read_parquet(f"{PROC}/more_sites_table.parquet")
    m = ExtraTreesRegressor(400, min_samples_leaf=2, random_state=0, n_jobs=-1)
    m.fit(d[FEATS].values, d.ET_closed_mm.values)
    return m


def load_production_model():
    """Load the saved production model (ExtraTrees on the 11 VIF-pruned features) and its
    feature list; fall back to the 14-feature model if the file is absent."""
    import joblib
    p = f"{PROC}/final_model.joblib"
    if os.path.exists(p):
        b = joblib.load(p)
        return b["model"], list(b["features"])
    return train_best(), FEATS


def open_catalog():
    return pystac_client.Client.open(STAC, modifier=pc.sign_inplace)


def open_gridmet():
    return {v: xr.open_dataset(GM.format(v=v))[nm] for v, nm in GVARS.items()}


def met_point(das, lat, lon, date):
    d = np.datetime64(pd.Timestamp(date).date())
    g = {v: float(das[v].sel(lat=lat, lon=lon, method="nearest").sel(day=d, method="nearest").values)
         for v in GVARS}
    tx, tn = g["tmmx"] - 273.15, g["tmmn"] - 273.15
    es = (es_kpa(tx) + es_kpa(tn)) / 2
    ea = (es_kpa(tn) * g["rmax"] / 100 + es_kpa(tx) * g["rmin"] / 100) / 2
    doy = pd.Timestamp(date).dayofyear
    return {"TA_ERA": (tx + tn) / 2, "VPD_ERA": max(es - ea, 0) * 10.0,
            "SW_IN_ERA": g["srad"], "WS_ERA": g["vs"], "ETo_mm": g["pet"],
            "DOY_sin": np.sin(2 * np.pi * doy / 365.25),
            "DOY_cos": np.cos(2 * np.pi * doy / 365.25)}


def indices(red, nir, grn, swr, lst):
    def safe(n, d, lo, hi):
        return np.clip(n / np.where(d > 0.02, d, np.nan), lo, hi)
    savi = np.clip(1.5 * (nir - red) / (nir + red + 0.5), -1, 1.5)
    lai = np.clip(-np.log((0.69 - np.clip(savi, 0, 0.685)) / 0.59) / 0.91, 0, 6)
    return {"NDVI": safe(nir - red, nir + red, -1, 1), "SAVI": savi,
            "EVI2": np.clip(2.5 * (nir - red) / (nir + 2.4 * red + 1), -1, 1.5),
            "NDWI": safe(nir - swr, nir + swr, -1, 1),
            "MNDWI": safe(grn - swr, grn + swr, -1, 1), "LAI": lai, "LST_K": lst}


def water_mask(idx):
    """Open water = wet AND not vegetated (high MNDWI, low NDVI). Marsh (green + wet)
    is kept; open bay/channel is flagged so the veg-trained model is not extrapolated
    over it."""
    return (idx["MNDWI"] > 0.0) & (idx["NDVI"] < 0.10)


def utm_epsg(lon, lat):
    return (32600 if lat >= 0 else 32700) + int((lon + 180) / 6) + 1


def pick_scene(cat, bbox, cx, cy):
    for cc in (8, 15, 25):
        items = [i for i in cat.search(collections=["landsat-c2-l2"], bbox=bbox,
                 datetime="2022-01-01/2023-12-31", query={"eo:cloud_cover": {"lt": cc}}).items()
                 if set(ASSETS) <= set(i.assets)
                 and i.bbox[0] <= cx <= i.bbox[2] and i.bbox[1] <= cy <= i.bbox[3]]
        if items:
            items.sort(key=lambda i: i.properties.get("eo:cloud_cover", 100))
            return items[0]
    return None


def predict_reserve(shp, model, cat, das, mask_water=True, feats=None):
    """Return a dict with the clipped ET map + coords + polygon + stats for one reserve."""
    feats = feats or FEATS
    g = gpd.read_file(shp).to_crs(4326)
    geom = g.geometry.union_all() if hasattr(g.geometry, "union_all") else g.geometry.unary_union
    b = g.total_bounds
    bbox = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    epsg = utm_epsg(cx, cy)
    it = pick_scene(cat, bbox, cx, cy)
    if it is None:
        return None
    date = pd.Timestamp(it.datetime).tz_convert("UTC").normalize()
    d0 = str(date.date())
    same = [i for i in cat.search(collections=["landsat-c2-l2"], bbox=bbox,
            datetime=f"{d0}/{d0}", query={"eo:cloud_cover": {"lt": 30}}).items()
            if set(ASSETS) <= set(i.assets)] or [it]
    st = stackstac.stack(same, assets=ASSETS, bounds_latlon=bbox, epsg=epsg,
                         resolution=30, chunksize=2048).compute()
    qa_t = np.nan_to_num(st.sel(band="qa_pixel").values, nan=1).astype("uint16")
    cloud_t = (qa_t & 0b111110) > 0
    bd = {}
    for a in ASSETS[:-1]:
        arr = st.sel(band=a).values.astype("float32")
        arr[cloud_t] = np.nan
        bd[a] = np.nanmean(arr, axis=0)
    for k in ("red", "nir08", "green", "swir16"):
        bd[k][bd[k] <= 0] = np.nan
    idx = indices(bd["red"], bd["nir08"], bd["green"], bd["swir16"], bd["lwir11"])
    met = met_point(das, cy, cx, date)

    H, W = idx["LST_K"].shape
    X = np.empty((H * W, len(feats)), dtype="float32")
    for j, f in enumerate(feats):
        X[:, j] = idx[f].ravel() if f in idx else met[f]

    xs, ys = st.x.values, st.y.values
    transform = Affine.translation(xs[0] - 15, ys[0] + 15) * Affine.scale(30, -30)
    poly = gpd.GeoSeries([geom], crs=4326).to_crs(epsg).iloc[0]
    inside = ~geometry_mask([poly], out_shape=(H, W), transform=transform, invert=False)

    water = water_mask(idx).ravel() if mask_water else np.zeros(H * W, bool)
    valid = np.isfinite(X).all(axis=1) & inside.ravel() & ~water
    et = np.full(H * W, np.nan, dtype="float32")
    if valid.sum():
        et[valid] = model.predict(X[valid])
    et = et.reshape(H, W)
    ninside = int(inside.sum())
    return dict(reserve=os.path.basename(os.path.dirname(shp)), date=d0, epsg=epsg,
                cloud=round(it.properties.get("eo:cloud_cover"), 1), et=et, x=xs, y=ys,
                poly=poly, bbox=bbox, pixels=int(valid.sum()),
                water_px=int((water & inside.ravel()).sum()), inside_px=ninside,
                mean_ET=round(float(np.nanmean(et)), 2) if valid.sum() else np.nan,
                min_ET=round(float(np.nanmin(et)), 2) if valid.sum() else np.nan,
                max_ET=round(float(np.nanmax(et)), 2) if valid.sum() else np.nan)


def save_outputs(r, outdir):
    os.makedirs(outdir, exist_ok=True)
    xa = xr.DataArray(r["et"], coords={"y": r["y"], "x": r["x"]}, dims=("y", "x"))
    xa.rio.write_crs(r["epsg"], inplace=True)
    xa.rio.to_raster(f"{outdir}/{r['reserve']}_ET_{r['date']}.tif")
    np.savez(f"{outdir}/{r['reserve']}_ET_{r['date']}.npz",
             et=r["et"], x=r["x"], y=r["y"], epsg=r["epsg"], date=r["date"], bbox=r["bbox"])
