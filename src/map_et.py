"""Apply the footprint-trained ET model to a full Landsat scene -> 30 m ET map.

Demonstration of the upscaling workflow: per-pixel LST/NDVI/MNDWI from one clear
Landsat overpass + gridMET met/ETo for that date -> ET at every pixel, over a
~24 km box covering the central Everglades cluster (Esm, Elm, EvM, TaS).

HONEST CAVEAT: leave-tower-out validation of this model is negative (R2<0), so
the ABSOLUTE ET values are not yet reliable for scaling to unmonitored pixels.
The map shows the workflow and the spatial PATTERN (which tracks NDVI/water), not
a validated product. Reported alongside the tower ET observed that day.
"""
import os
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "6")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
import numpy as np
import pandas as pd
import dask
import pyproj
import joblib
import planetary_computer as pc
import pystac_client
import stackstac
dask.config.set(scheduler="threads", num_workers=16)

R = "/anvil/scratch/x-jwang120/coastal-et"
OUT = f"{R}/data/processed"
FIG = f"{R}/figures"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
ASSETS = ["red", "nir08", "green", "swir16", "lwir11", "qa_pixel"]
# central Everglades box (lon_min, lat_min, lon_max, lat_max) ~24 km across
BBOX = [-80.85, 25.25, -80.55, 25.62]
EPSG = 32617
SITES = {"US-Esm": (25.4379, -80.5946), "US-Elm": (25.5519, -80.7826),
         "US-EvM": (25.3539, -80.3810), "US-TaS": (25.1908, -80.6391)}


def pick_scene(cat):
    items = [i for i in cat.search(collections=["landsat-c2-l2"], bbox=BBOX,
             datetime="2021-01-01/2021-12-31",
             query={"eo:cloud_cover": {"lt": 10}}).items()
             if set(ASSETS) <= set(i.assets)]
    items.sort(key=lambda i: i.properties.get("eo:cloud_cover", 100))
    return items[0] if items else None


def main():
    mdl = joblib.load(f"{OUT}/pixel_et_model.joblib")
    model, feats = mdl["model"], mdl["feats"]
    cat = pystac_client.Client.open(STAC, modifier=pc.sign_inplace)
    it = pick_scene(cat)
    if it is None:
        raise SystemExit("no clear scene")
    date = pd.Timestamp(it.datetime).tz_convert("UTC").normalize()
    print(f"scene {it.id}  {date.date()}  cloud {it.properties.get('eo:cloud_cover'):.1f}%")

    da = stackstac.stack([it], assets=ASSETS, bounds_latlon=BBOX, epsg=EPSG,
                         resolution=30, chunksize=1024).isel(time=0).compute()
    qa = da.sel(band="qa_pixel").values.astype("uint16")
    red = da.sel(band="red").values.astype("float32")
    nir = da.sel(band="nir08").values.astype("float32")
    grn = da.sel(band="green").values.astype("float32")
    swr = da.sel(band="swir16").values.astype("float32")
    lst = da.sel(band="lwir11").values.astype("float32")
    bad = (qa & 0b111110) > 0
    for v in (red, nir, grn, swr, lst):
        v[bad] = np.nan
    for a in (red, nir, grn, swr):
        a[a <= 0] = np.nan
    ndvi = (nir - red) / np.where((nir + red) > 0.02, nir + red, np.nan)
    ndvi[np.abs(ndvi) > 1] = np.nan
    mndwi = (grn - swr) / np.where((grn + swr) > 0.02, grn + swr, np.nan)
    mndwi[np.abs(mndwi) > 1] = np.nan

    # gridMET met + ETo for that date, nearest to the box centre (4 km -> ~uniform)
    latc, lonc = (BBOX[1] + BBOX[3]) / 2, (BBOX[0] + BBOX[2]) / 2
    met = {}
    # use US-Esm gridMET (inside the box) as the representative met for the date
    g = pd.read_parquet(f"{R}/data/interim/gridmet/US-Esm_gridmet.parquet")
    g.index = pd.to_datetime(g.index, utc=True)
    row = g.loc[g.index.normalize() == date]
    if len(row):
        r0 = row.iloc[0]
        met = {"TA_ERA": r0.get("TA_C"), "VPD_ERA": r0.get("VPD_kPa"),
               "SW_IN_ERA": r0.get("SRAD_Wm2"), "WS_ERA": r0.get("WS_ms"),
               "ETo_mm": r0.get("ETo_mm")}
    doy = date.dayofyear
    met["DOY_sin"] = np.sin(2 * np.pi * doy / 365.25)
    met["DOY_cos"] = np.cos(2 * np.pi * doy / 365.25)

    H, W = lst.shape
    X = np.empty((H * W, len(feats)), dtype="float32")
    layers = {"LST_K": lst, "NDVI": ndvi, "MNDWI": mndwi}
    for j, f in enumerate(feats):
        if f in layers:
            X[:, j] = layers[f].ravel()
        else:
            X[:, j] = met.get(f, np.nan)
    valid = np.isfinite(X).all(axis=1)
    et = np.full(H * W, np.nan, dtype="float32")
    if valid.sum():
        et[valid] = model.predict(X[valid])
    et = et.reshape(H, W)
    print(f"  predicted ET on {valid.sum():,}/{H*W:,} valid pixels  "
          f"(range {np.nanmin(et):.1f}-{np.nanmax(et):.1f} mm/d)")

    xs = da.x.values
    ys = da.y.values
    np.savez(f"{OUT}/et_map_{date.date()}.npz", et=et, x=xs, y=ys,
             ndvi=ndvi, mndwi=mndwi, date=str(date.date()), epsg=EPSG)

    # tower ET observed that day, for annotation
    etd = pd.read_parquet(f"{OUT}/daily_closed_et.parquet")
    etd.index = pd.to_datetime(etd.index, utc=True)
    obs = {}
    tf = pyproj.Transformer.from_crs(4326, EPSG, always_xy=True)
    for s, (la, lo) in SITES.items():
        if not (BBOX[0] <= lo <= BBOX[2] and BBOX[1] <= la <= BBOX[3]):
            continue                      # tower outside this scene box
        r = etd[(etd.SITE_ID == s) & (etd.index.normalize() == date)]
        obs[s] = (tf.transform(lo, la), float(r.ET_closed_mm.iloc[0]) if len(r) else np.nan)

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    ET_PAL = ["#DEC29B", "#E6CDA1", "#EDD9A6", "#F5E4A9", "#FFF4AD", "#C3E683",
              "#6BCC5C", "#3BB369", "#20998F", "#1C8691", "#16678A", "#114982", "#0B2C7A"]
    cmap = LinearSegmentedColormap.from_list("et", ET_PAL, N=256)
    cmap.set_bad("#e8e8e4")
    fig, ax = plt.subplots(figsize=(9, 10))
    fig.patch.set_facecolor("#fcfcfb")
    ext = [xs.min(), xs.max(), ys.min(), ys.max()]
    im = ax.imshow(np.ma.masked_invalid(et), origin="upper", extent=ext,
                   cmap=cmap, vmin=1, vmax=6)
    for s, ((tx, ty), v) in obs.items():
        ax.plot(tx, ty, marker="^", ms=13, color="#0b0b0b", mec="white", mew=1.5, zorder=5)
        lab = f"{s}\n{v:.1f} mm/d" if np.isfinite(v) else s
        ax.annotate(lab, (tx, ty), xytext=(9, 9), textcoords="offset points",
                    fontsize=9, fontweight="bold", color="#0b0b0b",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, extend="both")
    cb.set_label("modelled daily ET  (mm/day)", fontsize=11, color="#52514e")
    cb.outline.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"Footprint-trained ET map — central Everglades, {date.date()}",
                 loc="left", fontsize=13, fontweight="bold", color="#0b0b0b", pad=22)
    ax.text(0, 1.012,
            "30 m ET from Landsat LST/NDVI + gridMET; RF trained on footprint-weighted "
            "tower pixels.\n▲ tower (measured ET that day). Spatial pattern is "
            "indicative; absolute values are NOT leave-tower-out validated (R2<0).",
            transform=ax.transAxes, fontsize=8.5, color="#52514e", va="bottom")
    fig.subplots_adjust(top=0.88, bottom=0.02, left=0.02, right=0.99)
    fig.savefig(f"{FIG}/et_map_everglades.png", dpi=160, facecolor="#fcfcfb")
    print(f"  wrote {FIG}/et_map_everglades.png")


if __name__ == "__main__":
    main()
