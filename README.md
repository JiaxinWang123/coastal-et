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

## Repository layout
```
notebooks/   00 setup · 01 data overview · 02 model selection · 03 upscaling · 04 reserve prediction
src/         full analysis code; reserve_et.py is the portable pipeline notebook 04 imports
data/processed/   more_sites_table.parquet (833-record training table) · daily_closed_et.parquet
                  core_coastal_sites.csv · final_model.joblib (production model) · result CSVs
shp_predict/      NERR reserve boundary shapefiles (ACE, APA, GND, GTM, NIW, RKB, WKB)
figures/          generated figures + thumbnails
docs/             METHODS.md · CITATIONS.md
requirements.txt  pinned pip dependencies      environment.yml  conda spec
```
The 20 GB of raw imagery/flux data is **not** in this repo (size + AmeriFlux terms).
Notebooks 01 & 04 fetch imagery + meteorology live from public services instead.

## Reproduce
1. **`notebooks/00_environment_setup.ipynb`** → Run All. Installs `requirements.txt`
   (or build the conda env: `conda env create -f environment.yml`) and verifies the stack.
2. **`02` and `03`** → Run All. Fully **offline**; trains every model and draws every
   figure from the bundled table (~1–3 min each). This is the core reproducible result.
3. **`01` and `04`** need **outbound internet** (Microsoft Planetary Computer + gridMET).
   `04` predicts 30 m ET over the reserve polygons (~8–12 min).

Paths are derived from each notebook's own location — nothing to edit.

## Data & license
Processed table, model, and maps are released **CC-BY-4.0** (`LICENSE`). Underlying data
retain their own terms: AmeriFlux/FLUXNET (CC-BY-4.0), Landsat (USGS, public domain),
gridMET (public domain), ERA5 (Copernicus C3S). Sources & citations in `docs/CITATIONS.md`.

## Funding
NSF award **OAC-2118329** (I-GUIDE Institute).
