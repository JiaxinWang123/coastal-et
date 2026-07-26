"""Generate the runnable 04_spatial_prediction.ipynb (team voice).

The notebook RUNS the prediction pipeline inline (download -> indices -> met ->
ExtraTrees -> water mask -> clip) and builds the summary + panel figure live from the
results it just computed. Nothing is a pre-saved image. The reusable per-reserve
pipeline lives in the portable reserve_et module; the notebook drives it with visible
code (the loop, the summary, the figure, the GeoTIFF export are all here).
"""
import os
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook(); c = []
md = lambda t: c.append(new_markdown_cell(t))
co = lambda t: c.append(new_code_cell(t))

md("""# 04 · Spatial ET prediction over the reserve polygons

*Runnable notebook — it computes everything live. We apply our validated best model
(13-site ExtraTrees, leave-site R²≈0.72) across the NERR reserve boundaries in
`team2/shp_predict`, download the imagery and meteorology, predict ET at 30 m, mask open
water, clip to each reserve, and draw the maps from the results we just generated.*

Run it with the **Python (coastal-et)** kernel on a compute-backed session. Downloading
7 Landsat scenes + gridMET takes ~8–12 min.

Per reserve: pick a clear Landsat scene (2022–2023, same-date rows mosaicked) → compute
the 7 satellite indices per pixel → fetch gridMET point met (FAO-56 VPD) → ExtraTrees →
mask open water (`MNDWI>0 & NDVI<0.1`) → clip to the boundary.""")

co('''import os, sys, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# derive project root from the notebook location; import the portable pipeline
ROOT = os.environ.get("COASTAL_ET_ROOT")
if not ROOT or not os.path.isdir(os.path.join(ROOT, "data", "processed")):
    cand = os.path.dirname(os.getcwd())
    ROOT = cand if os.path.isdir(os.path.join(cand, "data", "processed")) \\
        else "/anvil/projects/x-ees260113/team2/coastal-et"
sys.path.insert(0, f"{ROOT}/src")
import reserve_et as RE
print("project root:", ROOT)
print("shapefiles  :", RE.SHP_DIR)''')

md("""## 1. Train the best model and open the data services

We load the saved **production model** (ExtraTrees on the 11 VIF-pruned features) and open
the Planetary Computer catalog and the gridMET OPeNDAP endpoints once (reused per reserve).""")

co('''model, feats = RE.load_production_model()   # ExtraTrees on the 11 VIF-pruned features
cat   = RE.open_catalog()
das   = RE.open_gridmet()
print(f"production model loaded ({len(feats)} features); catalog + gridMET ready")
print("features:", feats)''')

md("""## 2. Predict every reserve (this is the run)

The loop below is the actual computation — for each reserve boundary it downloads the
scene, builds the features, predicts ET, masks water, and clips to the polygon. It prints
progress as each reserve completes and keeps the results in `results`.""")

co('''shps = sorted(glob.glob(f"{RE.SHP_DIR}/*/*.shp"))
results = []
for shp in shps:
    name = os.path.basename(os.path.dirname(shp))
    r = RE.predict_reserve(shp, model, cat, das, mask_water=True, feats=feats)   # <- downloads + predicts
    if r is None:
        print(f"  {name:<4} no clear scene"); continue
    wf = 100 * r["water_px"] / max(r["inside_px"], 1)
    print(f"  {name:<4} {r['date']}  cloud {r['cloud']:>4}%  |  "
          f"{r['pixels']:>7,} land px  |  water {wf:>2.0f}% masked  |  "
          f"mean ET {r['mean_ET']} mm/day", flush=True)
    results.append(r)
print(f"\\ndone — {len(results)} reserves predicted")''')

md("""## 3. Summary table (built from what we just computed)""")

co('''summary = pd.DataFrame([{k: r[k] for k in
    ["reserve","date","cloud","mean_ET","min_ET","max_ET","pixels","water_px","inside_px","epsg"]}
    for r in results])
summary["water_pct"] = (100 * summary.water_px / summary.inside_px).round(0)
summary.to_csv(f"{RE.PROC}/reserve_maps/reserve_ET_summary.csv", index=False)  # ensure dir exists below
summary''')

md("""## 4. Panel figure — drawn here from the results

We build the multi-panel map inline (not a saved image): one clipped ET map per reserve,
on a shared colour scale, boundary outlined.""")

co('''PAL = ["#DEC29B","#EDD9A6","#FFF4AD","#C3E683","#6BCC5C","#3BB369","#20998F","#16678A","#114982"]
cmap = LinearSegmentedColormap.from_list("et", PAL, N=256); cmap.set_bad("#eeeeea")

n = len(results); ncol = 3; nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*3.4, nrow*3.2))
axes = np.atleast_1d(axes).ravel()
for ax in axes: ax.axis("off")
for ax, r in zip(axes, results):
    ext = [r["x"].min(), r["x"].max(), r["y"].min(), r["y"].max()]
    im = ax.imshow(np.ma.masked_invalid(r["et"]), origin="upper", extent=ext, cmap=cmap, vmin=1, vmax=6)
    gpd.GeoSeries([r["poly"]], crs=r["epsg"]).boundary.plot(ax=ax, color="#333", lw=0.6)
    ax.set_title(f"{r['reserve']}  {r['date']}\\nmean {r['mean_ET']} mm/d", fontsize=8.5, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([]); ax.axis("on")
    for sp in ax.spines.values(): sp.set_visible(False)
cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02, extend="both")
cb.set_label("predicted daily ET (mm/day)")
fig.suptitle("Predicted ET across NERR coastal-wetland reserves (open water masked)",
             fontsize=12, fontweight="bold")
plt.show()''')

md("""## 5. Save GeoTIFFs + arrays

Write each reserve's 30 m ET map as a GeoTIFF (UTM, open in QGIS/ArcGIS) plus an `.npz`.""")

co('''OUT = f"{RE.PROC}/reserve_maps"
os.makedirs(OUT, exist_ok=True)
for r in results:
    RE.save_outputs(r, OUT)
print(f"wrote {len(results)} GeoTIFFs + npz to {OUT}")
print("\\n".join(sorted(os.path.basename(f) for f in glob.glob(f"{OUT}/*.tif"))))''')

md("""## 6. Zoom on one reserve

Pick any reserve from the results for a larger view.""")

co('''pick = "WKB"      # change to any reserve code in the summary
r = next(x for x in results if x["reserve"] == pick)
fig, ax = plt.subplots(figsize=(7, 6))
ext = [r["x"].min(), r["x"].max(), r["y"].min(), r["y"].max()]
im = ax.imshow(np.ma.masked_invalid(r["et"]), origin="upper", extent=ext, cmap=cmap, vmin=1, vmax=6)
gpd.GeoSeries([r["poly"]], crs=r["epsg"]).boundary.plot(ax=ax, color="#333", lw=0.8)
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, extend="both")
cb.set_label("predicted daily ET (mm/day)"); cb.outline.set_visible(False)
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_visible(False)
ax.set_title(f"{r['reserve']} — predicted ET, {r['date']} (open water masked)", fontsize=12, fontweight="bold")
plt.show()''')

md("""### Caveats (also in `data/processed/reserve_maps/README.md`)
- **Resolution 30 m** from the optical indices; thermal LST is ~100 m resampled to 30 m;
  meteorology is one gridMET value per reserve/date (no sub-4 km variation).
- **Open water** is masked; marsh with standing water is kept.
- **Dates differ per reserve** (each uses its clearest scene) — fix the month for a fair
  cross-reserve comparison.
- These reserves are **new locations**; leave-site R²≈0.72 was on the 13 training
  wetlands, so treat absolute values as indicative and the spatial pattern as the product.""")

nb["cells"] = c
nb["metadata"] = {"kernelspec": {"display_name": "Python (coastal-et)", "language": "python", "name": "coastal-et"},
                  "language_info": {"name": "python"}}

OUTS = [
    "/anvil/scratch/x-jwang120/coastal-et/notebooks/04_spatial_prediction.ipynb",
    "/anvil/projects/x-ees260113/team2/coastal-et/notebooks/04_spatial_prediction.ipynb",
    "/home/x-jwang120/coastal-et/notebooks/04_spatial_prediction.ipynb",
]
for out in OUTS:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        nbf.write(nb, f)
    print("wrote", out, "-", len(c), "cells")
