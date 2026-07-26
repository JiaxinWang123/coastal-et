"""Generate 01_data_overview.ipynb (team voice) — data downloading, preprocessing,
and an EDA overview. Everything runs inline; the EDA is self-contained from the
processed tables, and the download cells show the live methods (STAC + gridMET).
"""
import os
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook(); c = []
md = lambda t: c.append(new_markdown_cell(t))
co = lambda t: c.append(new_code_cell(t))

md("""# 01 · Data — download, preprocessing, and overview

*Runnable notebook. We show where every input comes from, how we turn it into the
modelling table, and what the data actually looks like. The download cells run the live
methods (a STAC query, a gridMET point fetch); the overview is computed from the
processed tables. Kernel: **Python (coastal-et)**.*

Our target is **daily evapotranspiration (ET)** at coastal-wetland flux towers, predicted
from satellite + meteorology so it can be scaled in space. Three data streams feed it:

| Stream | Source | Resolution | Gives us |
|---|---|---|---|
| Flux towers | AmeriFlux / FLUXNET (ONEFlux) | 30-min → daily | measured LE → **ET** (the target) |
| Satellite | Landsat C2-L2 + Sentinel-2 (Planetary Computer) | 30 m / 10–20 m | NDVI, SAVI, EVI2, NDWI, MNDWI, LAI, **LST** |
| Meteorology | ERA5 (in FLUXNET) + gridMET (OPeNDAP) | tower / 4 km | TA, VPD, SW_in, WS, **ETo** |""")

md("""## Problem statement

Coastal wetlands are among the most productive and carbon-rich ecosystems on Earth, and
**evapotranspiration (ET)** — the water they return to the atmosphere — is central to their
water and energy balance. ET is measured directly only at a handful of **eddy-covariance
flux towers**, each representing a tiny footprint, so we cannot observe it across the vast,
inaccessible wetland landscape. **Our problem: can we learn the relationship between ET and
satellite + meteorological predictors at the towers, and use it to map daily ET at 30 m over
*unmonitored* coastal wetlands?** This matters for blue-carbon accounting, water-resource and
restoration management, and validating satellite ET products in a setting where they are
rarely tested. The core scientific question is whether such a model **transfers to a new,
unseen site** — which we test explicitly with leave-site-out cross-validation.""")

co('''import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

ROOT = os.environ.get("COASTAL_ET_ROOT")
if not ROOT or not os.path.isdir(os.path.join(ROOT, "data", "processed")):
    cand = os.path.dirname(os.getcwd())
    ROOT = cand if os.path.isdir(os.path.join(cand, "data", "processed")) \\
        else "/anvil/projects/x-ees260113/team2/coastal-et"
PROC = f"{ROOT}/data/processed"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.linewidth": 0.7, "figure.dpi": 120})
INK = "#1a1a1a"
print("project root:", ROOT)''')

md("""## 1. Study sites

We load the site metadata and keep the sites that made it into the modelling table
(a cloud-free overpass matched to a measured-ET day). These 13 span the Everglades plus
Atlantic/Gulf/Pacific coastal wetlands — the diversity that makes upscaling work.""")

co('''meta = pd.read_csv(f"{PROC}/core_coastal_sites.csv")
table = pd.read_parquet(f"{PROC}/more_sites_table.parquet")
model_sites = sorted(table.SITE_ID.unique())
inv = meta[meta.SITE_ID.isin(model_sites)][
    ["SITE_ID","SITE_NAME","STATE","IGBP","LAT","LON","KOEPPEN","MAT","MAP"]].copy()
# attach sample count + mean ET from the table
agg = table.groupby("SITE_ID").agg(n_overpass=("ET_closed_mm","size"),
                                    mean_ET=("ET_closed_mm","mean")).round(2)
inv = inv.merge(agg, on="SITE_ID").sort_values("LAT")
print(f"{len(inv)} modelling sites across {inv.STATE.nunique()} states")
inv''')

md("""### Where they are""")
co('''fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(inv.LON, inv.LAT, c=inv.mean_ET, s=90, cmap="YlGnBu",
                edgecolor="#333", linewidth=0.6, zorder=3, vmin=2, vmax=5)
for _, r in inv.iterrows():
    ax.annotate(r.SITE_ID, (r.LON, r.LAT), xytext=(4, 4), textcoords="offset points", fontsize=6.5)
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_title("Coastal-wetland flux towers (colour = mean daily ET)", fontsize=10, fontweight="bold")
cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02); cb.set_label("mean ET (mm/day)")
ax.grid(alpha=0.25, zorder=0)
plt.show()''')

md("""## 2. Downloading the data (live methods)

We don't re-download everything here (the full extraction is a cluster job), but the
cells below run the **actual access methods** so you can see how each stream is pulled.""")

md("""**2a. Flux towers — AmeriFlux/FLUXNET.** We use the ONEFlux FLUXNET product (half-hourly
LE, H, Rn, G, plus ERA5 met). Each site's page and metadata:""")
co('''print("example AmeriFlux site pages:")
for _, r in inv.head(4).iterrows():
    print(f"  {r.SITE_ID}  {r.SITE_NAME:<40} https://ameriflux.lbl.gov/sites/siteinfo/{r.SITE_ID}")''')

md("""**2b. Satellite — Planetary Computer STAC.** A live query for clear Landsat scenes over
one site (this hits the network; needs a compute-backed session).""")
co('''import planetary_computer as pc, pystac_client
site = inv.iloc[len(inv)//2]
cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                modifier=pc.sign_inplace)
items = list(cat.search(collections=["landsat-c2-l2"],
             bbox=[site.LON-0.05, site.LAT-0.05, site.LON+0.05, site.LAT+0.05],
             datetime="2022-01-01/2022-12-31",
             query={"eo:cloud_cover": {"lt": 10}}).items())
print(f"{site.SITE_ID}: {len(items)} clear Landsat scenes in 2022")
if items:
    it = sorted(items, key=lambda i: i.properties['eo:cloud_cover'])[0]
    print("  clearest:", it.id, f"{it.properties['eo:cloud_cover']:.1f}% cloud")
    print("  assets used: red, nir08, green, swir16, lwir11 (thermal), qa_pixel")''')

md("""**2c. Meteorology — gridMET via OPeNDAP.** A live point fetch (no big download): a few
days of reference ET and temperature at one site.""")
co('''import xarray as xr
GM = "http://thredds.northwestknowledge.net:8080/thredds/dodsC/agg_met_{v}_1979_CurrentYear_CONUS.nc"
eto = xr.open_dataset(GM.format(v="pet"))["daily_mean_reference_evapotranspiration_grass"]
pt = eto.sel(lat=site.LAT, lon=site.LON, method="nearest").sel(
     day=slice("2022-06-01", "2022-06-07"))
print(f"gridMET ETo at {site.SITE_ID}, first week of June 2022 (mm/day):")
print(np.round(pt.values, 2))''')

md("""## 3. Preprocessing

**3a. Energy-balance closure → ET.** Eddy covariance under-closes the surface energy
balance (H+LE < Rn−G). We correct LE (Bowen-ratio / ONEFlux `LE_CORR`), convert to ET
with a temperature-dependent latent heat, and aggregate to daily (≥80% coverage). The
correction is not cosmetic — it moves ET by 10–30%, so we keep both.""")
co('''et = pd.read_parquet(f"{PROC}/daily_closed_et.parquet")
clo = (et[et.ET_closed_mm.notna()].groupby("SITE_ID")
       .agg(ET_open=("ET_open_mm","mean"), ET_closed=("ET_closed_mm","mean"),
            closure=("CLOSURE","first")).round(2))
clo["uplift_%"] = ((clo.ET_closed/clo.ET_open - 1)*100).round(1)
clo = clo[clo.index.isin(model_sites)].sort_values("uplift_%")

fig, ax = plt.subplots(figsize=(6.4, 3.6))
y = np.arange(len(clo))
ax.barh(y, clo["uplift_%"], color="#4C72B0", height=0.7, zorder=3)
ax.set_yticks(y); ax.set_yticklabels(clo.index, fontsize=7)
ax.set_xlabel("ET increase from energy-balance closure (%)")
for sp in ("top","right"): ax.spines[sp].set_visible(False)
ax.set_title("Closure correction raises ET by 10–30%", fontsize=10, fontweight="bold")
ax.grid(axis="x", alpha=0.25, zorder=0)
plt.show()
clo''')

md("""**3b. Satellite indices — computed per pixel, then averaged.** Because indices are
nonlinear, we compute them on each pixel before window-averaging (ratio-of-means is biased
over mixed marsh/water). The exact formulas we use:

```
NDVI  = (NIR − Red) / (NIR + Red)
SAVI  = 1.5·(NIR − Red) / (NIR + Red + 0.5)
EVI2  = 2.5·(NIR − Red) / (NIR + 2.4·Red + 1)
NDWI  = (NIR − SWIR) / (NIR + SWIR)
MNDWI = (Green − SWIR) / (Green + SWIR)
LAI   = −ln((0.69 − SAVI) / 0.59) / 0.91     (DisALEXI/TSEB relation)
LST   = Landsat thermal band (K)             (Sentinel-2 has no thermal)
```
These are the 7 satellite features. LST is Landsat-only, which is why Landsat is
indispensable and Sentinel-2 only densifies the optical record.""")

md("""**3c. Flux-footprint weighting (Kljun et al. 2015).** The tower sees a
footprint-integrated flux, so we weight satellite pixels by their 2-D footprint
contribution (Obukhov length from u*/H/T, σv≈1.9u*, literature tower heights). This is a
heavy precompute (`src/footprint_climatology.py`); the modelling table already carries the
footprint-weighted features.""")

md("""## 4. Data overview (the modelling table)

Everything below is computed live from `more_sites_table.parquet` — the analysis-ready
matches of footprint-weighted satellite + meteorology to measured daily ET.""")

co('''SAT = ["LAI","EVI2","SAVI","NDVI","NDWI","MNDWI","LST_K"]
MET = ["TA_ERA","VPD_ERA","SW_IN_ERA","WS_ERA","ETo_mm","DOY_sin","DOY_cos"]
print(f"{len(table)} overpass matches | {table.SITE_ID.nunique()} sites | "
      f"years {int(table.year.min())}–{int(table.year.max())}")
print(f"target: ET_closed_mm  (mean {table.ET_closed_mm.mean():.2f}, "
      f"range {table.ET_closed_mm.min():.2f}–{table.ET_closed_mm.max():.2f} mm/day)")
table[["SITE_ID","year"] + SAT + MET + ["ET_closed_mm"]].describe().round(2).T[["mean","std","min","max"]]''')

md("""### ET distribution by site""")
co('''order = table.groupby("SITE_ID").ET_closed_mm.median().sort_values().index
data = [table[table.SITE_ID == s].ET_closed_mm.values for s in order]
fig, ax = plt.subplots(figsize=(7.4, 3.8))
bp = ax.boxplot(data, vert=True, patch_artist=True, widths=0.6,
                medianprops=dict(color="#222"), flierprops=dict(ms=2, alpha=0.3))
for p in bp["boxes"]:
    p.set(facecolor="#6BCC5C", edgecolor="#333", linewidth=0.6)
ax.set_xticks(range(1, len(order)+1)); ax.set_xticklabels(order, rotation=45, ha="right", fontsize=7)
ax.set_ylabel("daily ET (mm/day)")
for sp in ("top","right"): ax.spines[sp].set_visible(False)
ax.set_title("Measured daily ET by site", fontsize=10, fontweight="bold")
plt.show()''')

md("""### Overpass coverage — matches per site per year""")
co('''piv = table.pivot_table(index="SITE_ID", columns="year", values="ET_closed_mm",
                        aggfunc="size", fill_value=0)
fig, ax = plt.subplots(figsize=(6.2, 4.2))
im = ax.imshow(piv.values, cmap="YlGnBu", aspect="auto")
ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns.astype(int))
ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=7)
for i in range(len(piv.index)):
    for j in range(len(piv.columns)):
        v = piv.values[i, j]
        if v: ax.text(j, i, int(v), ha="center", va="center", fontsize=6,
                      color="white" if v > piv.values.max()*0.5 else "#333")
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("matches")
ax.set_title("Cloud-free overpass matches per site-year", fontsize=10, fontweight="bold")
plt.show()''')

md("""### Feature distributions""")
co('''feats = SAT[:6] + ["LST_K","TA_ERA","VPD_ERA","SW_IN_ERA","WS_ERA","ETo_mm"]
fig, axes = plt.subplots(3, 4, figsize=(9, 6))
for ax, f in zip(axes.ravel(), feats):
    ax.hist(table[f].dropna(), bins=30, color="#4C72B0", alpha=0.85)
    ax.set_title(f, fontsize=8); ax.tick_params(labelsize=6)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
for ax in axes.ravel()[len(feats):]: ax.axis("off")
fig.suptitle("Predictor distributions across all site-days", fontsize=11, fontweight="bold")
fig.tight_layout()
plt.show()''')

md("""### How features relate to ET""")
co('''cols = SAT + ["TA_ERA","VPD_ERA","SW_IN_ERA","WS_ERA","ETo_mm","ET_closed_mm"]
corr = table[cols].corr()["ET_closed_mm"].drop("ET_closed_mm").sort_values()
fig, ax = plt.subplots(figsize=(5.2, 4.2))
col = ["#55A868" if f in SAT else "#4C72B0" for f in corr.index]
ax.barh(np.arange(len(corr)), corr.values, color=col, height=0.7, zorder=3)
ax.set_yticks(np.arange(len(corr))); ax.set_yticklabels(corr.index, fontsize=7.5)
ax.axvline(0, color="#888", lw=0.8)
ax.set_xlabel("Pearson r with daily ET")
for sp in ("top","right"): ax.spines[sp].set_visible(False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc="#55A868", label="satellite"), Patch(fc="#4C72B0", label="meteorology")],
          frameon=False, fontsize=7, loc="lower right")
ax.set_title("Correlation of each predictor with ET", fontsize=10, fontweight="bold")
plt.show()''')

md("""### Takeaways
- **13 diverse coastal wetlands**, ~830 cloud-free overpass matches, 2022–2023 focus.
- **Energy-balance closure** raises ET 10–30% — larger than most model differences.
- Meteorology (**ETo**, VPD, SW_in) correlates most strongly with ET; among satellite
  features the **water/moisture indices and LST** lead, raw greenness is weaker — the
  same signal the model's feature importance shows later.

Next: **`02_model_selection.ipynb`** trains models on exactly this table.""")

md("""## Data sources, licensing & citations

All inputs are open data; our derived products are released **CC-BY-4.0**.

- **AmeriFlux / FLUXNET (ONEFlux)** — flux ET (target). CC-BY-4.0. Pastorello et al. (2020),
  *Scientific Data* 7:225; cite each site's data DOI.
- **Landsat Collection-2 L2** — reflectance + surface temperature. USGS, public domain.
- **Sentinel-2 L2A** — optical reflectance. Copernicus/ESA, free & open.
- **Microsoft Planetary Computer** — STAC access to the imagery.
- **gridMET** — reference ET & meteorology. Abatzoglou (2013), *Int. J. Climatol.* 33:121–131.
- **ERA5** — meteorology. Hersbach et al. (2020), *QJRMS* 146:1999–2049 (Copernicus C3S).
- **Kljun et al. (2015)** — flux-footprint model (robustness test). *GMD* 8:3695–3713.
- **scikit-learn** — Pedregosa et al. (2011), *JMLR* 12:2825–2830 (BSD-3).

Full list and the dataset description are in `docs/CITATIONS.md` and
`data/processed/DATASET_README.md`.

**How to run this project:** see `00_environment_setup.ipynb` — register the shared
`Python (coastal-et)` kernel, then run `01 → 02 → 03 → 04`. Notebooks 02 and 03 are fully
offline (just the processed table); 01 and 04 fetch imagery/meteorology live.""")

nb["cells"] = c
nb["metadata"] = {"kernelspec": {"display_name": "Python (coastal-et)", "language": "python", "name": "coastal-et"},
                  "language_info": {"name": "python"}}

OUTS = [
    "/anvil/scratch/x-jwang120/coastal-et/notebooks/01_data_overview.ipynb",
    "/anvil/projects/x-ees260113/team2/coastal-et/notebooks/01_data_overview.ipynb",
    "/home/x-jwang120/coastal-et/notebooks/01_data_overview.ipynb",
]
for out in OUTS:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        nbf.write(nb, f)
    print("wrote", out, "-", len(c), "cells")
