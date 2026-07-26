# Data sources, licences, and citations

## Data
- **AmeriFlux / FLUXNET (ONEFlux)** — flux-tower LE/H/ET and micrometeorology.
  Licence: AmeriFlux CC-BY-4.0. Cite each site's data DOI and: Pastorello et al. (2020),
  *The FLUXNET2015 dataset and the ONEFlux processing pipeline*, Scientific Data 7:225.
- **Landsat Collection-2 Level-2** — surface reflectance + surface temperature.
  Courtesy of the U.S. Geological Survey; public domain.
- **Sentinel-2 L2A** — surface reflectance (optical fusion). Copernicus/ESA; free & open.
- **Microsoft Planetary Computer** — STAC access to Landsat/Sentinel-2. Terms: MPC data use.
- **gridMET** — 4 km reference ET & meteorology. Abatzoglou (2013), *Development of gridded
  surface meteorological data*, Int. J. Climatology 33:121–131. Public domain (via NW Knowledge / OPeNDAP).
- **ERA5** — Hersbach et al. (2020), *The ERA5 global reanalysis*, QJRMS 146:1999–2049.
  Copernicus Climate Change Service (C3S); Copernicus licence.
- **NERR reserve boundaries** (`shp_predict`) — NOAA National Estuarine Research Reserve System.

## Methods / software
- **Flux footprint:** Kljun, Calanca, Rotach, Schmid (2015), *A simple two-dimensional
  parameterisation for flux-footprint prediction (FFP)*, Geosci. Model Dev. 8:3695–3713.
- **scikit-learn** — Pedregosa et al. (2011), JMLR 12:2825–2830 (BSD-3).
- Also: stackstac, pystac-client, xarray, rioxarray, geopandas, statsmodels, xgboost, lightgbm.

## This work
- Derived products (`more_sites_table.parquet`, `final_model.joblib`, reserve ET maps):
  released **CC-BY-4.0**. Cite as: Team 2, I-GUIDE Summer School 2026, "Upscaling
  coastal-wetland evapotranspiration from flux towers to satellite."
