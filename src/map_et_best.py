"""Spatial ET prediction over the Florida Everglades wetlands with our BEST model.

Unlike the old map_et.py (3-feature pixel model, leave-tower-out R2<0), this uses the
validated 13-site ExtraTrees model (leave-site R2~0.72) and computes ALL 7 satellite
indices per pixel with the exact training-time formulas, so the map is consistent with
what the model learned.

  best model : ExtraTrees on the 13-site table (14 features)
  scene      : one clear Landsat C2-L2 overpass over the Everglades box
  indices    : NDVI, SAVI, EVI2, NDWI, MNDWI, LAI, LST_K  (per pixel)
  meteorology: gridMET for the scene date (uniform over the box; VPD kPa->hPa)
"""
import os
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "6")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
import numpy as np
import pandas as pd
import dask
import pyproj
import planetary_computer as pc
import pystac_client
import stackstac
from sklearn.ensemble import ExtraTreesRegressor
dask.config.set(scheduler="threads", num_workers=16)

R = "/anvil/scratch/x-jwang120/coastal-et"
OUT = f"{R}/data/processed"
FIG = f"{R}/figures"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
ASSETS = ["red", "nir08", "green", "swir16", "lwir11", "qa_pixel"]
FEATS = ["LAI", "EVI2", "SAVI", "NDVI", "NDWI", "MNDWI", "LST_K",
         "TA_ERA", "VPD_ERA", "SW_IN_ERA", "WS_ERA", "ETo_mm", "DOY_sin", "DOY_cos"]
# Everglades marsh cluster — sized to sit inside a single Landsat scene swath.
# (US-Skr mangrove is one Landsat path west, so it is mapped separately, not here.)
BBOX = [-80.92, 25.18, -80.34, 25.66]
CLUSTER = (-80.60, 25.40)          # scene must cover this to map the marshes
EPSG = 32617
SITES = {"US-Esm": (25.4379, -80.5946), "US-Elm": (25.5519, -80.7826),
         "US-EvM": (25.3539, -80.3810), "US-TaS": (25.1908, -80.6391)}


def train_best():
    d = pd.read_parquet(f"{OUT}/more_sites_table.parquet")
    m = ExtraTreesRegressor(400, min_samples_leaf=2, random_state=0, n_jobs=-1)
    m.fit(d[FEATS].values, d.ET_closed_mm.values)
    print(f"trained ExtraTrees on {len(d)} samples, {d.SITE_ID.nunique()} sites")
    return m


def pick_scene(cat):
    cx, cy = CLUSTER
    items = [i for i in cat.search(collections=["landsat-c2-l2"], bbox=BBOX,
             datetime="2022-01-01/2023-12-31",
             query={"eo:cloud_cover": {"lt": 8}}).items()
             if set(ASSETS) <= set(i.assets)
             and i.bbox[0] <= cx <= i.bbox[2] and i.bbox[1] <= cy <= i.bbox[3]]
    items.sort(key=lambda i: i.properties.get("eo:cloud_cover", 100))
    return items[0] if items else None


def indices_per_pixel(red, nir, grn, swr, lst):
    """Exact training-time formulas (more_sites.indices_from_bands), vectorised."""
    def safe(num, den, lo, hi, thr=0.02):
        out = num / np.where(den > thr, den, np.nan)
        return np.clip(out, lo, hi)
    ndvi = safe(nir - red, nir + red, -1, 1)
    savi = np.clip(1.5 * (nir - red) / (nir + red + 0.5), -1, 1.5)
    evi2 = np.clip(2.5 * (nir - red) / (nir + 2.4 * red + 1), -1, 1.5)
    ndwi = safe(nir - swr, nir + swr, -1, 1)
    mndwi = safe(grn - swr, grn + swr, -1, 1)
    savi_c = np.clip(savi, 0, 0.685)
    lai = np.clip(-np.log((0.69 - savi_c) / 0.59) / 0.91, 0, 6)
    return {"NDVI": ndvi, "SAVI": savi, "EVI2": evi2, "NDWI": ndwi,
            "MNDWI": mndwi, "LAI": lai, "LST_K": lst}


def gridmet_met(date):
    g = pd.read_parquet(f"{R}/data/interim/gridmet/US-Esm_gridmet.parquet")
    g.index = pd.to_datetime(g.index, utc=True)
    row = g.loc[g.index.normalize() == date]
    if not len(row):
        raise SystemExit(f"no gridMET for {date.date()}")
    r0 = row.iloc[0]
    doy = date.dayofyear
    return {"TA_ERA": r0["TA_C"], "VPD_ERA": r0["VPD_kPa"] * 10.0,   # kPa -> hPa
            "SW_IN_ERA": r0["SRAD_Wm2"], "WS_ERA": r0["WS_ms"], "ETo_mm": r0["ETo_mm"],
            "DOY_sin": np.sin(2 * np.pi * doy / 365.25),
            "DOY_cos": np.cos(2 * np.pi * doy / 365.25)}


def main():
    model = train_best()
    cat = pystac_client.Client.open(STAC, modifier=pc.sign_inplace)
    it = pick_scene(cat)
    if it is None:
        raise SystemExit("no clear scene in window")
    date = pd.Timestamp(it.datetime).tz_convert("UTC").normalize()
    print(f"scene {it.id}  {date.date()}  cloud {it.properties.get('eo:cloud_cover'):.1f}%")

    da = stackstac.stack([it], assets=ASSETS, bounds_latlon=BBOX, epsg=EPSG,
                         resolution=30, chunksize=2048).isel(time=0).compute()
    b = {a: da.sel(band=a).values.astype("float32") for a in ASSETS[:-1]}
    qa = da.sel(band="qa_pixel").values.astype("uint16")
    bad = (qa & 0b111110) > 0
    for v in b.values():
        v[bad] = np.nan
    for k in ("red", "nir08", "green", "swir16"):
        b[k][b[k] <= 0] = np.nan
    idx = indices_per_pixel(b["red"], b["nir08"], b["green"], b["swir16"], b["lwir11"])
    met = gridmet_met(date)
    print("  met:", {k: round(float(v), 2) for k, v in met.items() if k != "DOY_sin" and k != "DOY_cos"})

    H, W = idx["LST_K"].shape
    X = np.empty((H * W, len(FEATS)), dtype="float32")
    for j, f in enumerate(FEATS):
        X[:, j] = idx[f].ravel() if f in idx else met[f]
    valid = np.isfinite(X).all(axis=1)
    et = np.full(H * W, np.nan, dtype="float32")
    et[valid] = model.predict(X[valid])
    et = et.reshape(H, W)
    print(f"  predicted {valid.sum():,}/{H*W:,} pixels | ET {np.nanmin(et):.1f}-{np.nanmax(et):.1f} "
          f"mean {np.nanmean(et):.2f} mm/d")

    xs, ys = da.x.values, da.y.values
    np.savez(f"{OUT}/et_map_best_{date.date()}.npz", et=et, x=xs, y=ys,
             date=str(date.date()), epsg=EPSG, bbox=BBOX)

    # tower ET observed that day
    etd = pd.read_parquet(f"{OUT}/daily_closed_et.parquet")
    etd.index = pd.to_datetime(etd.index, utc=True)
    tf = pyproj.Transformer.from_crs(4326, EPSG, always_xy=True)
    obs = {}
    for s, (la, lo) in SITES.items():
        if not (BBOX[0] <= lo <= BBOX[2] and BBOX[1] <= la <= BBOX[3]):
            continue
        r = etd[(etd.SITE_ID == s) & (etd.index.normalize() == date)]
        obs[s] = (tf.transform(lo, la), float(r.ET_closed_mm.iloc[0]) if len(r) else np.nan)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    PAL = ["#DEC29B", "#EDD9A6", "#FFF4AD", "#C3E683", "#6BCC5C", "#3BB369",
           "#20998F", "#16678A", "#114982", "#0B2C7A"]
    cmap = LinearSegmentedColormap.from_list("et", PAL, N=256); cmap.set_bad("#e8e8e4")
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#fcfcfb")
    ext = [xs.min(), xs.max(), ys.min(), ys.max()]
    im = ax.imshow(np.ma.masked_invalid(et), origin="upper", extent=ext, cmap=cmap,
                   vmin=1, vmax=6)
    for s, ((tx, ty), v) in obs.items():
        ax.plot(tx, ty, "^", ms=12, color="#0b0b0b", mec="white", mew=1.4, zorder=5)
        lab = f"{s}\n{v:.1f}" if np.isfinite(v) else s
        ax.annotate(lab, (tx, ty), xytext=(8, 8), textcoords="offset points", fontsize=8.5,
                    fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, extend="both")
    cb.set_label("predicted daily ET (mm/day)", fontsize=11); cb.outline.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"Everglades ET — validated 13-site ExtraTrees model, {date.date()}",
                 loc="left", fontsize=13, fontweight="bold", pad=14)
    fig.savefig(f"{FIG}/et_map_best_everglades.png", dpi=160, facecolor="#fcfcfb", bbox_inches="tight")
    print(f"  wrote {FIG}/et_map_best_everglades.png")


if __name__ == "__main__":
    main()
