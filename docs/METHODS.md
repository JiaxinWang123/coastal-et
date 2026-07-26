# Coastal-wetland ET: methods, data download → final prediction

Goal: estimate **daily evapotranspiration (ET)** at coastal-wetland flux towers from
satellite + meteorology, and scale it spatially to unmonitored wetlands.

## 1. Data download (all idempotent: skip-if-exists, `--force` to redo)
- **Flux towers** — AmeriFlux/FLUXNET ONEFlux product via the AmeriFlux API
  (`download_fluxnet.py`): half-hourly LE, H, Rn, G, u\*, wind, and bundled ERA5 met.
- **Satellite** — Landsat C2-L2 (30 m, optical+thermal) and Sentinel-2 L2A (10–20 m,
  optical) from the Microsoft **Planetary Computer** STAC (`download_satellite.py`,
  `extract_tower_point.py`, `extract_pixels.py`).
- **Meteorology** — ERA5 (in the FLUXNET product) + **gridMET** 4 km via OPeNDAP /
  annual NetCDF (`download_gridmet.py`), giving reference ET (ETo).

## 2. Flux-tower ET (the target)
`daily_closed_et.py`, `process_flux.py`:
- **Energy-balance closure** — eddy covariance under-closes (H+LE < Rn−G); corrected
  with the Bowen-ratio method / ONEFlux `LE_CORR`.
- ET = LE / λ (λ temperature-dependent latent heat); aggregated to **daily** requiring
  ≥80 % half-hourly coverage. Closure raises ET **10–30 %**; both open and closed kept.
- Target is **measured** closed ET — never gap-filled.

## 3. Satellite predictors
`extract_tower_point.py`, `more_sites.py`:
- Cloud masking (Landsat QA_PIXEL / Sentinel-2 SCL) + minimum valid-pixel fraction.
- **Indices computed per pixel before averaging** (ratios are nonlinear):
  NDVI, SAVI, EVI2, **NDWI** (NIR−SWIR), MNDWI, **LAI** (from SAVI, DisALEXI relation),
  **LST** (Landsat thermal only — Sentinel-2 has no thermal band).
- **Sentinel-2 fused with Landsat** (cross-calibrated on common bands B3/B4/B8/B11) to
  densify the optical record between Landsat overpasses.
- Tower feature = **500 m window mean** (90/250 m also stored).
- **Flux footprint (Kljun 2015)** computed for all 13 sites and tested as an alternative
  feature support — see §8; it does not beat the window.

## 4. Meteorology features
TA_ERA, VPD_ERA, SW_IN_ERA, WS_ERA (ERA5, merged/gap-filled), ETo_mm (gridMET), and
**DOY_sin / DOY_cos** — a cyclic encoding of day-of-year (season), so Dec 31 ≈ Jan 1.

## 5. Analysis-ready table
Each cloud-free overpass day is matched to its satellite + met features and the measured
closed ET → **833 samples (site-days) across 13 sites, 14 features**
(`more_sites_table.parquet`).

## 6. Model selection
`compare_13sites.py`, `final_model_selection.py`:
- **Model zoo:** Ridge, ElasticNet, PLS, kNN, SVR, Gaussian process, RandomForest,
  **ExtraTrees**, GradientBoosting, HistGBM, XGBoost, LightGBM (deep MLPs tried early,
  overfit at n≈833).
- **Cross-validation — grouped, three schemes** (`evaluate()`):
  1. random 10-fold (predict at monitored sites),
  2. leave-year-out (unseen years),
  3. **leave-site-out** (LeaveOneGroupOut by SITE_ID — predict a *completely unmonitored
     tower*; this is the **headline** metric for spatial upscaling).
  Pooled R² (predictions concatenated across folds, one R²). No fixed hold-out set —
  with 13 sites, leave-one-site-out *is* the test.

## 7. Feature analysis & selection
- **Permutation importance (leave-site-out):** ETo dominates; among satellite, water
  (NDWI) + thermal (LST) lead; raw greenness negligible.
- **Group ablation** (`feature_group_ablation.py`): met-only (0.716) ≈ full (0.715);
  greenness-only ≈ noise (−0.00); water+LST (0.28).
- **Multicollinearity / VIF** (`multicollinearity.py`): greenness indices are near-
  duplicates (EVI2 VIF≈8500); iterative elimination drops EVI2/SAVI/NDWI → **11 features**.
- **AIC/BIC stepwise** on the linear model (`aic_bic_selection.py`): BIC drops *all*
  greenness, keeping water+thermal+met — same verdict (AIC/BIC used only where valid,
  i.e. the linear model, not the tree ensembles).
- **Hyperparameter search** (`tune_model.py`): no gain (default ExtraTrees already at the
  ceiling). Reframing to predict Kc=ET/ETo was worse.

## 8. Feature support: window vs flux footprint
`footprint_climatology.py`, `extract_pixels.py`, `build_footprint_table.py`:
- Kljun 2015 footprint climatology for **all 13** sites; per-pixel indices footprint-
  weighted. Leave-site R² **0.707 (footprint) vs 0.722 (500 m window)** — footprint is
  marginally *worse*. Window sizes 90/250/500 m all give ~0.72. **The model is
  insensitive to feature support**, so the 500 m window stays production and the 30 m
  prediction is empirically justified (see `FOOTPRINT_ANALYSIS.md`).

## 9. Final production model
**ExtraTrees on the 11 VIF-pruned features** (LAI, NDVI, MNDWI, LST + 7 met), refit on
all 833 samples → `final_model.joblib` (+ `final_model.json`). Leave-site-out
**R² ≈ 0.72, MAE ≈ 0.8 mm/day**.

## 10. Spatial prediction (upscaling)
`reserve_et.py`, `map_reserves.py`:
- For each target polygon (7 NERR reserves) pull a clear Landsat scene (same-date rows
  mosaicked), compute the 7 indices **per 30 m pixel**, fetch gridMET point met (FAO-56
  VPD), predict per pixel, **mask open water** (MNDWI>0 & NDVI<0.1), clip to the boundary.
- Outputs: 30 m ET **GeoTIFFs**, a reserve panel, and a CONUS locator figure.

## 11. Headline findings
- **Leave-site-out is positive (R²≈0.72) only with a *diverse* 13-site network**; 5
  spectrally-similar Everglades sites fail (R²<0). The bottleneck is **training-set
  diversity**, not the model, features, or physics.
- **Daily wetland ET is demand-limited** — meteorology (ETo) carries the temporal skill;
  satellite adds little to *point* daily ET but is **essential for the 30 m spatial
  pattern** (met is 4 km/uniform). Footprint weighting doesn't change this.
- Predictions are on a **30 m grid**; effective resolution ≈ tower/feature support, which
  the footprint test shows is not a limiting factor here.
