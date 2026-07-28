# Coastal-wetland evapotranspiration: reproducible ML notebooks

Estimating **daily evapotranspiration (ET)** at coastal eddy-covariance flux towers and
upscaling it to **30 m maps** over National Estuarine Research Reserve (NERR) wetlands.
I-GUIDE Summer School 2026, Team 2.

## Headline result
12 models × 3 cross-validation schemes; four independent feature-selection methods
(permutation importance, group ablation, VIF, AIC/BIC) agree on ~6–7 inputs. Production
model: **ExtraTrees on 11 VIF-pruned features, leave-site-out R² ≈ 0.72**. Meteorology
(especially reference ET) carries the signal; satellite indices add spatial texture. The
ceiling is **training-site diversity**, not model choice.

## Start here
**`notebooks/coastal_wetland_ET_upscaling.ipynb`** runs the whole story end-to-end — problem → data → 12-model
comparison → feature selection → spatial-upscaling validation → live 30 m reserve
prediction → conclusions. The modular `00`–`04` notebooks remain as the detailed components.

## Repository layout
```
notebooks/   coastal_wetland_ET_upscaling (full workflow) · 00 setup · 01 data · 02 models · 03 upscaling · 04 prediction
src/         full analysis code; reserve_et.py is the portable pipeline notebook 04 imports
data/processed/   more_sites_table.parquet (833-record training table) · daily_closed_et.parquet
                  core_coastal_sites.csv · final_model.joblib (production model) · result CSVs
shp_predict/      NERR reserve boundary shapefiles (ACE, APA, GND, GTM, NIW, RKB, WKB)
figures/          generated figures + thumbnails
docs/             METHODS.md · CITATIONS.md
requirements.txt  pinned pip dependencies      environment.yml  conda spec
```
The 20 GB of raw imagery/flux data is **not** in this repo (size + AmeriFlux terms) and is
**not needed** — the processed table and the pre-computed 30 m maps here drive the whole
workflow. Live imagery/meteorology is fetched only if you opt into `RECOMPUTE` or predict a
new area.

## Reproduce
Open **`notebooks/coastal_wetland_ET_upscaling.ipynb`** and **Run All** — it runs the whole
workflow in ~10 minutes and is self-bootstrapping:
- Its first cell **auto-installs** `requirements.txt` if the stack is missing, resolves the
  project root wherever the repo is mounted, and fetches the analysis table from the
  published dataset if absent. So it runs as-is on the **I-GUIDE JupyterHub** (which clones
  this repo) — no manual setup.
- **No raw-data download.** The raw→features work is already baked into
  `more_sites_table.parquet` (Parts 1–3), and Part 4's 30 m reserve maps are **pre-computed**
  in `data/processed/reserve_maps/*.npz` and loaded by default.
- To regenerate the reserve maps from scratch instead, set **`RECOMPUTE = True`** in Part 4
  (re-downloads Landsat + gridMET, ~10 min, needs internet).

Prefer the modular pieces? `notebooks/00`–`04` run each stage individually. Paths self-resolve
from the notebook location — nothing to edit.

## Data availability
The analysis table and the 30 m ET GeoTIFFs are published on the **I-GUIDE Platform**:
<https://platform.i-guide.io/datasets/a0a5736a-4a53-4fb5-b20d-33d4e6019992>
(direct: `https://storage.i-guide.io/datasets/a0a5736a-4a53-4fb5-b20d-33d4e6019992/coastal_et_dataset.zip`).
This repo includes the analysis table and the reserve-map arrays (`.npz`); the GeoTIFFs live
in that published dataset. The ~20 GB of raw imagery/flux is not redistributed (size +
AmeriFlux terms) — it is a rapid download from AmeriFlux, USGS Landsat, gridMET, and ERA5.

## Data & license
Processed table, model, and maps are released **CC-BY-4.0** (`LICENSE`). Underlying data
retain their own terms: AmeriFlux/FLUXNET (CC-BY-4.0), Landsat (USGS, public domain),
gridMET (public domain), ERA5 (Copernicus C3S). Sources & citations in `docs/CITATIONS.md`.

## Funding
NSF award **OAC-2118329** (I-GUIDE Institute).

## Data & model documentation (I-DET cards)
Documented per the I-GUIDE Data Ethics Toolkit (I-DET v2.0):
1. [Data Card 1](https://docs.google.com/document/d/1t0te7AOK5PyBxMpze5rV4AOQWvHYr4CP/edit)
2. [Data Card 2](https://docs.google.com/document/d/1OO5PLDC0ouaLdYmnn0hmJx3R9gn6Q3_h/edit)
3. [Model Card](https://docs.google.com/document/d/1bEjWCd567tLpJUw1Pc7j_HoDt3mYpgXK/edit)
4. [Card Changes and Suggestions Statement](https://docs.google.com/document/d/1PcD-crwiCG3tBV9mZKzYh9_ouigZuoie/edit)
