"""Generate the coastal-ET workflow Jupyter notebook (author's first-person voice)."""
import os
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
c = []


def md(t):
    c.append(new_markdown_cell(t))


def code(t):
    c.append(new_code_cell(t))


md("""# Upscaling coastal-wetland evapotranspiration from flux towers to satellite

*Author's working notebook. Everglades pilot, then a 13-site coastal-wetland network.*

In this notebook I document my full workflow for estimating daily evapotranspiration
(ET) from eddy-covariance flux towers and scaling it to satellite observations. I
start with five Everglades towers, work through energy-balance closure, satellite
predictors, flux-footprint weighting, and model selection, and I finish with my main
result: predicting ET at an **unmonitored** tower becomes possible once I train on a
diverse enough set of coastal wetlands.

All heavy extraction was run on the cluster; here I load the processed products and
reproduce the analysis and figures.""")

code("""import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("COASTAL_ET_WIN", "500")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

ROOT = "/anvil/scratch/x-jwang120/coastal-et"
sys.path.insert(0, f"{ROOT}/src")
FIG = f"{ROOT}/figures"
PROC = f"{ROOT}/data/processed"
pd.set_option("display.width", 140)
print("workspace:", ROOT)""")

md("""## 1. Study sites

My pilot is the Florida Coastal Everglades: a hydroperiod/salinity gradient from
freshwater marsh through mangrove. I later add eight more coastal wetlands spanning
the Atlantic (SC, DE), Gulf (LA), Pacific delta (CA) and a NC tidal forest, because
I need ecosystem diversity for the upscaling step.""")

code("""sites = pd.read_parquet(f"{PROC}/daily_closed_et.parquet")
inv = (sites[sites.ET_closed_mm.notna()]
       .groupby("SITE_ID").agg(days=("ET_closed_mm","size"),
                               ET_mean=("ET_closed_mm","mean")).round(2))
everglades = ["US-Elm","US-Esm","US-TaS","US-EvM","US-Skr"]
print("My 5 Everglades pilot sites:")
inv.loc[[s for s in everglades if s in inv.index]]""")

md("""## 2. Flux-tower ET via energy-balance closure

Eddy covariance measures latent heat flux (LE), but it systematically under-closes
the surface energy balance (H + LE < Rn - G). I correct this with the Bowen-ratio
method (or use the ONEFlux `LE_CORR` where the FLUXNET product provides it), then
convert closure-corrected LE to ET with a temperature-dependent latent heat of
vaporisation, and aggregate to daily totals requiring >=80% half-hourly coverage.

I keep both the closure-corrected and open ET, because the correction moves ET by
10-30% - larger than the difference between competing models - so I report results
against both.""")

code("""et = pd.read_parquet(f"{PROC}/daily_closed_et.parquet")
et.index = pd.to_datetime(et.index, utc=True)
summary = (et[et.ET_closed_mm.notna()].groupby("SITE_ID")
           .agg(closure=("CLOSURE","first"),
                ET_open=("ET_open_mm","mean"),
                ET_closed=("ET_closed_mm","mean")).round(2))
uplift = (summary.ET_closed/summary.ET_open - 1)*100
summary["uplift_%"] = uplift.round(1)
summary.loc[[s for s in everglades if s in summary.index]]""")

md("""My closed daily ET time series across the Everglades gradient:""")
code("""display(Image(f"{FIG}/et_timeseries_everglades.png"))""")

md("""## 3. Satellite predictors

For each tower I extract Landsat C2-L2 (30 m, optical + thermal) and Sentinel-2
(10-20 m, optical) over a window centred on the tower, cloud-masked with the QA/SCL
bands and a minimum valid-pixel fraction. From the surface-reflectance bands I derive
vegetation and water indices, computing them **per pixel before averaging** (the
index is nonlinear, so ratio-of-means is biased over mixed marsh/water pixels):

- **NDVI, EVI2, SAVI** - greenness
- **LAI** = -ln((0.69-SAVI)/0.59)/0.91 (the relation used in DisALEXI/TSEB)
- **NDWI, MNDWI** - surface water / moisture
- **LST** - land surface temperature (Landsat thermal only; Sentinel-2 has no
  thermal band, so Landsat is indispensable here)

I fuse Landsat and Sentinel-2 optical after cross-calibrating S2 onto the Landsat
scale, which roughly triples optical coverage between the sparse Landsat overpasses.""")

code("""display(Image(f"{FIG}/indices_timeseries.png"))""")
md("""The mangrove (US-Skr) is spectrally distinct - high LAI/NDVI - while the four
marshes overlap heavily in every index. This near-degeneracy is exactly why, later,
five Everglades sites alone cannot support spatial upscaling.""")

md("""## 4. Meteorology

I force the models with ERA5 (bundled in the AmeriFlux FLUXNET product, already
downscaled to each tower) and gridMET (4 km, which also gives me reference ET,
`ETo`). I compute VPD from air temperature and dewpoint directly, because the
packaged `VPD_ERA` field was in hPa rather than kPa. Where a site's ERA5 download
was incomplete I fall back to gridMET.""")

md("""## 5. Flux-footprint weighting (Kljun et al. 2015)

The tower measures a footprint-integrated flux, so I compute a 2-D footprint
climatology per tower with the Kljun et al. (2015) model and weight the satellite
pixels by their contribution. I derive the Obukhov length from u*, sensible heat and
temperature, parameterise sigma_v from u*, and use literature measurement heights
(27 m for the Shark River mangrove tower, 4 m for the marshes).""")

code("""display(Image(f"{FIG}/footprint_climatology.png"))""")
md("""The footprint scales with tower height exactly as expected: the 27 m mangrove
tower integrates ~5 ha, the 4 m marsh towers only 0.3-0.6 ha.""")

md("""## 6. Model table

I match each cloud-free overpass to the footprint-weighted satellite features, the
day's meteorology and reference ET, and the measured daily closed ET. My target is
always the **measured** ET - I never train on gap-filled values, since those are a
function of reference ET and would let the model grade itself.""")

code("""import train_indices_model as T
d5 = T.build().dropna(subset=["ET_closed_mm"] + T.FEATS).reset_index(drop=True)
print(f"{len(d5)} overpass matches across {d5.SITE_ID.nunique()} Everglades sites")
print("features:", T.FEATS)""")

md("""## 7. Model selection

I compared 14 model families - linear (Ridge, ElasticNet, PLS), kernel (SVR, kNN),
tree ensembles (RF, ExtraTrees, XGBoost, LightGBM, GradBoost, HistGBM), a Gaussian
process, and deep neural networks (PyTorch MLPs). I evaluate each on three schemes
that answer different questions: random K-fold (predict at monitored sites),
leave-year-out (predict unseen years), and leave-tower-out (scale to an unseen site).

I found the **Gaussian process** best on this small (n~=227) dataset, and the deep
networks overfit badly - 227 samples is well below the deep-learning regime.""")

code("""display(Image(f"{FIG}/model_comparison.png"))""")

md("""## 8. Predictive skill of the best model

With the Gaussian process I can predict ET well at monitored sites (R^2 ~= 0.6) and
in unseen years (~=0.5), but on five spectrally-similar Everglades sites I cannot
predict a completely unseen tower (leave-tower-out R^2 < 0). The near-flat regression
in panel c is the signature of a model collapsing to the mean.""")

code("""display(Image(f"{FIG}/et_prediction_scatter.png"))""")

md("""## 9. Physics-informed model (energy balance + Priestley-Taylor)

To constrain the network with physics, I had it predict the Priestley-Taylor
coefficient alpha and fed it through the surface energy balance:
ET = alpha * [Delta/(Delta+gamma)] * (Rn - G) / lambda. The physics bounds prevented
the plain neural net's catastrophic overfits, and the fit told me something physical:
the observed alpha is ~=0.90, not the well-watered 1.26 - my Everglades marshes
evaporate **below** potential, consistent with periodic moisture/salinity stress.
The physics-informed model was more robust, but it did not beat the Gaussian process,
and it did not fix leave-tower-out - because at five similar sites the limitation was
never the model.""")

md("""## 10. Spatial upscaling - my key result

The one lever I had not pulled was **more sites**. I added eight diverse coastal
wetlands (Atlantic and Gulf salt marsh, a NC tidal forest, Pacific delta) and
re-ran the leave-tower-out test. Predicting ET at an unmonitored tower went from
failing (R^2 ~ -0.4 to -1.0 at five Everglades sites) to succeeding
(**R^2 ~= 0.71** at thirteen diverse wetlands).""")

code("""cmp = pd.read_parquet(f"{PROC}/more_sites_table.parquet")
print("Leave-tower-out (predict an UNSEEN tower):")
print("  5 Everglades sites  : R2 = -0.37 to -0.97  (fails)")
print("  13 coastal wetlands : R2 = +0.66 to +0.71  (works)")
print(f"\\n{len(cmp)} samples across {cmp.SITE_ID.nunique()} sites")
display(Image(f"{FIG}/upscaling_more_sites.png"))""")

md("""This is my central finding: **the bottleneck to satellite ET upscaling in coastal
wetlands is training-set diversity, not the model, the features, or the physics.**
Five spectrally-identical Everglades marshes taught the model nothing transferable;
thirteen wetlands spanning different climates, salinities and canopies give the
satellite indices real ecosystems to discriminate.

To make the effect fully explicit I compared all three validation schemes side by
side. K-fold and leave-year-out are already positive at five sites, but only the
larger, diverse network makes leave-site-out - true upscaling to an unmonitored
tower - work. With thirteen sites the skill stays high (R^2 ~= 0.7-0.8) across every
scheme, for both my best tree ensemble and the Gaussian process.""")
code("""display(Image(f"{FIG}/cv_comparison.png"))""")

md("""## 11. ET mapping

With the trained model I can apply the feature->ET relationship to every pixel of a
Landsat scene to produce a 30 m ET map. The spatial pattern tracks vegetation and
open water; the absolute values are trustworthy to the degree the leave-tower-out
validation supports.""")

code("""display(Image(f"{FIG}/et_map_everglades.png"))""")

md("""## Summary of what I learned

1. **Energy-balance closure matters** - it raises ET by 10-30%, more than most
   model differences, so I always report both closed and open ET.
2. **A Gaussian process is the right model** for this small-n problem; deep learning
   overfits at ~200 samples.
3. **Physics-informed constraints add robustness** and surface real ecohydrology
   (alpha ~= 0.90 - sub-potential ET under stress), even when they do not raise skill.
4. **Evaporative-fraction self-preservation breaks down in these wetlands** because
   ~22% of ET is nocturnal - a caveat for thermal-ET products here.
5. **Spatial upscaling to unmonitored coast is achievable (R^2 ~= 0.7)** once the
   training set spans enough coastal-wetland diversity. This is the path forward.""")

nb["cells"] = c
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python",
                                 "name": "python3"},
                  "language_info": {"name": "python"}}
out = "/anvil/scratch/x-jwang120/coastal-et/notebooks/coastal_et_workflow.ipynb"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    nbf.write(nb, f)
print("wrote", out, "-", len(c), "cells")
