"""Two I-GUIDE card thumbnails:
  - dataset_thumbnail.png : study-area map (13 towers) + a crisp 30 m reserve ET map,
    i.e. "point flux towers -> gridded ET product".
  - notebook_thumbnail.png: leave-site-out predicted-vs-observed ET (the headline result).
Square-ish 1200x900 cards, ET colormap, readable at small size.
"""
import os, glob
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_absolute_error

P = "/anvil/projects/x-ees260113/team2/coastal-et"
OUT = f"{P}/figures/thumbnails"; os.makedirs(OUT, exist_ok=True)
PAL = ["#DEC29B","#EDD9A6","#FFF4AD","#C3E683","#6BCC5C","#3BB369","#20998F","#16678A","#114982"]
CMAP = LinearSegmentedColormap.from_list("et", PAL, N=256); CMAP.set_bad("#e9e9e4")

FEATS = ["LAI","NDVI","MNDWI","LST_K","TA_ERA","VPD_ERA","SW_IN_ERA","WS_ERA","ETo_mm","DOY_sin","DOY_cos"]


# ---------------- dataset thumbnail ----------------
def dataset_thumb():
    sites = pd.read_csv(f"{P}/data/processed/core_coastal_sites.csv")
    d = pd.read_parquet(f"{P}/data/processed/more_sites_table.parquet")
    dist = pd.read_csv(f"{P}/data/processed/us_sites_coastal_distance.csv")
    coord = pd.concat([sites[["SITE_ID","LAT","LON"]],
                       dist.rename(columns={c:c.upper() for c in dist.columns})[["SITE_ID","LAT","LON"]]
                       ]).drop_duplicates("SITE_ID").set_index("SITE_ID")
    ids = sorted(d.SITE_ID.unique())
    la = [coord.loc[s,"LAT"] for s in ids if s in coord.index]
    lo = [coord.loc[s,"LON"] for s in ids if s in coord.index]

    # a crisp reserve ET map for the right panel (Grand Bay — big, clean marsh)
    z = np.load(sorted(glob.glob(f"{P}/data/processed/reserve_maps/GND_*.npz"))[0], allow_pickle=True)
    et, x, y = z["et"], z["x"], z["y"]

    try:
        import urllib.request, io, zipfile
        cache = f"{P}/data/raw/cb_2022_us_state_20m.shp"
        if not os.path.exists(cache):
            data = urllib.request.urlopen(
                "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_state_20m.zip", timeout=90).read()
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            zipfile.ZipFile(io.BytesIO(data)).extractall(os.path.dirname(cache))
        st = gpd.read_file(cache); st = st[~st.STUSPS.isin({"AK","HI","PR","GU","MP","AS","VI"})].to_crs(4326)
    except Exception as e:
        print("states failed:", e); st = None

    fig = plt.figure(figsize=(12, 9)); fig.patch.set_facecolor("white")
    axm = fig.add_axes([0.02, 0.09, 0.60, 0.82])
    if st is not None:
        st.plot(ax=axm, color="#f3f3ef", edgecolor="#cfcfcf", lw=0.5)
    axm.scatter(lo, la, s=95, c="#c0392b", edgecolor="white", lw=1.4, zorder=5)
    axm.set_xlim(-125, -73); axm.set_ylim(23.5, 41.5)
    axm.set_xticks([]); axm.set_yticks([])
    for s in axm.spines.values(): s.set_visible(False)
    axm.set_title("13 coastal flux towers", fontsize=17, fontweight="bold", color="#333", pad=6)

    axr = fig.add_axes([0.635, 0.16, 0.345, 0.66])
    im = axr.imshow(np.ma.masked_invalid(et), origin="upper",
                    extent=[x.min(), x.max(), y.min(), y.max()], cmap=CMAP, vmin=1, vmax=6)
    axr.set_xticks([]); axr.set_yticks([])
    for s in axr.spines.values(): s.set_edgecolor("#c0392b"); s.set_linewidth(1.6)
    axr.set_title("30 m ET map\n(per reserve)", fontsize=15, fontweight="bold", color="#333", pad=6)
    cax = fig.add_axes([0.655, 0.115, 0.30, 0.022])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both")
    cb.set_label("daily ET (mm/day)", fontsize=11); cb.outline.set_visible(False)

    fig.text(0.5, 0.955, "Coastal-wetland evapotranspiration dataset",
             ha="center", fontsize=20, fontweight="bold", color="#16678A")
    fig.text(0.31, 0.045, "point measurements", ha="center", fontsize=12, color="#666", style="italic")
    fig.text(0.81, 0.045, "gridded 30 m product", ha="center", fontsize=12, color="#666", style="italic")
    fig.text(0.62, 0.50, "→", ha="center", fontsize=40, color="#999")
    p = f"{OUT}/dataset_thumbnail.png"
    fig.savefig(p, dpi=110, facecolor="white"); plt.close(fig); print("wrote", p)


# ---------------- notebook thumbnail ----------------
def notebook_thumb():
    d = pd.read_parquet(f"{P}/data/processed/more_sites_table.parquet")
    sites = sorted(d.SITE_ID.unique())
    yt, yp = [], []
    for s in sites:
        tr, te = d[d.SITE_ID != s], d[d.SITE_ID == s]
        if len(te) < 5: continue
        m = ExtraTreesRegressor(600, max_features=1.0, min_samples_leaf=2, random_state=0, n_jobs=-1)
        m.fit(tr[FEATS].values, tr.ET_closed_mm.values)
        yp.append(m.predict(te[FEATS].values)); yt.append(te.ET_closed_mm.values)
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    r2 = r2_score(yt, yp); mae = mean_absolute_error(yt, yp)

    fig, ax = plt.subplots(figsize=(11, 9)); fig.patch.set_facecolor("white")
    hb = ax.hexbin(yt, yp, gridsize=42, mincnt=1, cmap="YlGnBu", bins="log", linewidths=0.15)
    lim = [0, max(yt.max(), yp.max()) * 1.03]
    ax.plot(lim, lim, "--", color="#c0392b", lw=2, zorder=5)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("observed ET  (mm/day)", fontsize=15)
    ax.set_ylabel("predicted ET  (mm/day)", fontsize=15)
    ax.tick_params(labelsize=12)
    ax.text(0.05, 0.93, f"leave-site-out\n$R^2$ = {r2:.2f}   MAE = {mae:.2f} mm/d",
            transform=ax.transAxes, fontsize=17, fontweight="bold", va="top", color="#16678A")
    ax.set_title("Predicting daily wetland ET at unseen sites",
                 fontsize=19, fontweight="bold", color="#333", pad=12)
    cb = fig.colorbar(hb, ax=ax, fraction=0.045, pad=0.02); cb.set_label("count", fontsize=12)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    p = f"{OUT}/notebook_thumbnail.png"
    fig.savefig(p, dpi=110, facecolor="white", bbox_inches="tight"); plt.close(fig)
    print("wrote", p, f"(R2={r2:.3f})")


if __name__ == "__main__":
    dataset_thumb()
    notebook_thumb()
