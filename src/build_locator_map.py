"""Final review figure: a CONUS locator map with the 7 NERR reserves marked, and
leader-line callouts to each reserve's zoomed ET-prediction map.
"""
import os, glob, io, zipfile, urllib.request
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import ConnectionPatch

R = "/anvil/scratch/x-jwang120/coastal-et"
MAPS = f"{R}/data/processed/reserve_maps"
SHP = "/anvil/projects/x-ees260113/team2/shp_predict"
FIG = f"{R}/figures/reserve_maps"
PAL = ["#DEC29B","#EDD9A6","#FFF4AD","#C3E683","#6BCC5C","#3BB369","#20998F","#16678A","#114982"]
CMAP = LinearSegmentedColormap.from_list("et", PAL, N=256); CMAP.set_bad("#eeeeea")


def conus_states():
    """Census CONUS state boundaries (download+cache); fall back to Natural Earth USA."""
    cache = f"{R}/data/raw/cb_2022_us_state_20m.shp"
    if not os.path.exists(cache):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        url = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_state_20m.zip"
        try:
            data = urllib.request.urlopen(url, timeout=120).read()
            zipfile.ZipFile(io.BytesIO(data)).extractall(os.path.dirname(cache))
        except Exception as e:
            print("states download failed, using Natural Earth:", e)
            ne = glob.glob(os.path.dirname(gpd.__file__)+"/**/naturalearth_lowres*.shp", recursive=True)
            g = gpd.read_file(ne[0]); return g[g.name == "United States of America"]
    g = gpd.read_file(cache)
    drop = {"AK","HI","PR","GU","MP","AS","VI"}
    return g[~g.STUSPS.isin(drop)].to_crs(4326)


def load_reserves():
    summ = pd.read_csv(f"{MAPS}/reserve_ET_summary.csv").set_index("reserve")
    out = {}
    for f in sorted(glob.glob(f"{MAPS}/*.npz")):
        name = os.path.basename(f).split("_")[0]
        z = np.load(f, allow_pickle=True)
        bbox = z["bbox"]; epsg = int(z["epsg"])
        shp = glob.glob(f"{SHP}/{name}/*.shp")[0]
        poly = gpd.read_file(shp).to_crs(epsg)
        out[name] = dict(et=z["et"], x=z["x"], y=z["y"], epsg=epsg,
                         lon=(bbox[0]+bbox[2])/2, lat=(bbox[1]+bbox[3])/2,
                         poly=poly, mean=summ.loc[name, "mean_ET"], date=str(z["date"]))
    return out


def main():
    states = conus_states()
    res = load_reserves()

    fig = plt.figure(figsize=(15, 9.5))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.24, 0.16, 0.52, 0.76])          # CONUS map
    states.boundary.plot(ax=ax, color="#b9b9b9", lw=0.5, zorder=1)
    states.plot(ax=ax, color="#f4f4f2", edgecolor="#c9c9c9", lw=0.4, zorder=0)
    ax.set_xlim(-107, -66); ax.set_ylim(23.5, 45)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title("Predicted evapotranspiration across NERR coastal-wetland reserves",
                 fontsize=14, fontweight="bold", pad=10)

    # inset slots (figure fraction): 3 down the left, 4 down the right, colorbar
    # gets the clear bottom-centre strip. Ordered top->bottom by latitude.
    slots = {
        "GND": [0.005, 0.66, 0.205, 0.24], "WKB": [0.005, 0.37, 0.205, 0.24],
        "APA": [0.005, 0.08, 0.205, 0.24],
        "NIW": [0.79, 0.765, 0.205, 0.185], "ACE": [0.79, 0.525, 0.205, 0.185],
        "GTM": [0.79, 0.285, 0.205, 0.185], "RKB": [0.79, 0.045, 0.205, 0.185],
    }
    im = None
    for name, r in res.items():
        # dot + label on the CONUS map
        ax.plot(r["lon"], r["lat"], "o", ms=7, mfc="#c0392b", mec="white", mew=1.2, zorder=6)
        # inset
        axin = fig.add_axes(slots[name])
        ext = [r["x"].min(), r["x"].max(), r["y"].min(), r["y"].max()]
        im = axin.imshow(np.ma.masked_invalid(r["et"]), origin="upper", extent=ext,
                         cmap=CMAP, vmin=1, vmax=6)
        r["poly"].boundary.plot(ax=axin, color="#333", lw=0.7)
        axin.set_xticks([]); axin.set_yticks([])
        for sp in axin.spines.values():
            sp.set_edgecolor("#c0392b"); sp.set_linewidth(1.1)
        axin.set_title(f"{name} · {r['mean']:.1f} mm/d\n{r['date']}", fontsize=8.6,
                       fontweight="bold", pad=2)
        # leader line from the map dot to the inset's inner edge
        onleft = slots[name][0] < 0.3
        xB, yB = (1.0 if onleft else 0.0), 0.5
        con = ConnectionPatch(xyA=(r["lon"], r["lat"]), coordsA=ax.transData,
                              xyB=(xB, yB), coordsB=axin.transAxes,
                              color="#c0392b", lw=0.9, alpha=0.8, zorder=5)
        fig.add_artist(con)

    # shared colorbar in the clear bottom-centre strip (below the CONUS map)
    cax = fig.add_axes([0.37, 0.085, 0.26, 0.018])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both")
    cb.set_label("predicted daily ET (mm/day)", fontsize=10); cb.outline.set_visible(False)
    fig.text(0.5, 0.035, "30 m ET from the validated 13-site ExtraTrees model; open water masked. "
             "Each callout uses its clearest 2022–2023 Landsat scene.",
             ha="center", fontsize=8.5, color="#555")

    out = f"{FIG}/reserve_ET_locator.png"
    fig.savefig(out, dpi=180, facecolor="white", bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
