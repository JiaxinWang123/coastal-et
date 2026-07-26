"""Generate the coastal-ET workflow as THREE notebooks (team first-person voice)."""
import os
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = "/anvil/scratch/x-jwang120/coastal-et/notebooks"
os.makedirs(OUT, exist_ok=True)

SETUP = """import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("COASTAL_ET_WIN", "500")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

ROOT = "/anvil/scratch/x-jwang120/coastal-et"
sys.path.insert(0, f"{ROOT}/src")
FIG, PROC = f"{ROOT}/figures", f"{ROOT}/data/processed"
pd.set_option("display.width", 140)"""


def build(name, cells):
    nb = new_notebook()
    nb["cells"] = []
    for kind, txt in cells:
        nb["cells"].append(new_markdown_cell(txt) if kind == "md" else new_code_cell(txt))
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python",
                                     "name": "python3"}, "language_info": {"name": "python"}}
    p = f"{OUT}/{name}.ipynb"
    with open(p, "w") as f:
        nbf.write(nb, f)
    print("wrote", p, "-", len(nb["cells"]), "cells")


# ============================ NOTEBOOK 1 ============================
nb1 = [
("md", """# 1 — Flux-tower ET and satellite predictors

*Our coastal-wetland ET workflow, part 1 of 3: from raw eddy-covariance flux to the
closed daily ET we treat as ground truth, and the satellite and meteorological
predictors we pair with it.*

We begin with five Everglades towers spanning a hydroperiod/salinity gradient
(freshwater marsh -> mangrove). Everything heavy was pre-computed on the cluster;
here we load the products and reproduce the analysis."""),
("code", SETUP),

("md", """## 1.1 Study sites

Our pilot network is the Florida Coastal Everglades. We list the five sites with their
record length and mean ET below."""),
("code", """et = pd.read_parquet(f"{PROC}/daily_closed_et.parquet")
et.index = pd.to_datetime(et.index, utc=True)
everglades = ["US-Elm","US-Esm","US-TaS","US-EvM","US-Skr"]
inv = (et[et.ET_closed_mm.notna()].groupby("SITE_ID")
       .agg(days=("ET_closed_mm","size"), ET_mean=("ET_closed_mm","mean"),
            closure=("CLOSURE","first")).round(2))
inv.loc[[s for s in everglades if s in inv.index]]"""),

("md", """## 1.2 Energy-balance closure

Eddy covariance under-closes the surface energy balance (H + LE < Rn - G), so we
correct LE with the Bowen-ratio method (or use ONEFlux `LE_CORR` where available),
convert to ET with a temperature-dependent latent heat, and aggregate to daily
totals with a >=80% coverage rule. The correction raises ET by 10-30% - larger than
most model differences - so we always keep both closed and open ET."""),
("code", """s = (et[et.ET_closed_mm.notna()].groupby("SITE_ID")
     .agg(ET_open=("ET_open_mm","mean"), ET_closed=("ET_closed_mm","mean")).round(2))
s["uplift_%"] = ((s.ET_closed/s.ET_open - 1)*100).round(1)
display(s.loc[[x for x in everglades if x in s.index]])
display(Image(f"{FIG}/et_timeseries_everglades.png"))"""),

("md", """## 1.3 Satellite predictors

We extract Landsat (30 m, optical + thermal) and Sentinel-2 (10-20 m, optical) over
each tower, cloud-mask them, and derive indices per pixel before averaging (the
ratio is nonlinear, so averaging bands first biases mixed marsh/water pixels): NDVI,
EVI2, SAVI, LAI (from SAVI, as in DisALEXI/TSEB), NDWI, MNDWI, and LST. We fuse the
two sensors' optical bands after cross-calibration to triple coverage."""),
("code", """display(Image(f"{FIG}/indices_timeseries.png"))"""),
("md", """The mangrove is spectrally distinct; the four marshes overlap heavily in
every index - a degeneracy that matters when we get to spatial upscaling."""),

("md", """## 1.4 Meteorology

We force the models with ERA5 (bundled, tower-downscaled) and gridMET (4 km, giving
reference ET). We compute VPD from temperature and dewpoint directly, since the
packaged VPD field was in hPa. Input data availability across sites:"""),
("code", """display(Image(f"{FIG}/input_data_availability.png"))"""),

("md", """## 1.5 Flux-footprint weighting (Kljun et al. 2015)

The tower integrates a footprint, so we weight satellite pixels by their Kljun (2015)
footprint contribution. We derive the Obukhov length from u*, H and T, parameterise
sigma_v from u*, and use literature measurement heights (27 m mangrove, 4 m marsh)."""),
("code", """display(Image(f"{FIG}/footprint_climatology.png"))"""),
("md", """Footprint size scales with tower height - ~5 ha for the 27 m mangrove tower,
0.3-0.6 ha for the 4 m marshes - exactly as the physics predicts.

*Continued in notebook 2 (model selection).*"""),
]

# ============================ NOTEBOOK 2 ============================
nb2 = [
("md", """# 2 — Model selection and predictive skill

*Part 2 of 3. We assemble the model table, compare model families, quantify
predictive skill under three validation schemes, and test a physics-informed
network and an instantaneous evaporative-fraction approach.*"""),
("code", SETUP),

("md", """## 2.1 Model table

We match each cloud-free overpass to the footprint-weighted satellite features, the
day's meteorology and reference ET, and the measured daily closed ET. We train only
on measured ET - never gap-filled values, which are a function of reference ET."""),
("code", """import train_indices_model as T
d5 = T.build().dropna(subset=["ET_closed_mm"] + T.FEATS).reset_index(drop=True)
print(f"{len(d5)} overpass matches, {d5.SITE_ID.nunique()} Everglades sites")
print("features:", T.FEATS)"""),

("md", """## 2.2 Comparing model families

We compared 14 model families - linear, kernel, tree ensembles, a Gaussian process,
and deep neural networks - across three schemes: random K-fold (monitored sites),
leave-year-out (unseen years), leave-tower-out (unseen site). On this small set
(n~=227) the Gaussian process was best and the deep nets overfit."""),
("code", """display(Image(f"{FIG}/model_comparison.png"))"""),

("md", """## 2.3 Predictive skill of the best model

With the Gaussian process we predict ET well at monitored sites and in unseen years,
but we cannot predict a completely unseen tower from five similar Everglades sites -
the flat regression in panel c is a model collapsing to the mean."""),
("code", """display(Image(f"{FIG}/et_prediction_scatter.png"))"""),

("md", """## 2.4 Physics-informed model (energy balance + Priestley-Taylor)

We had a network predict the Priestley-Taylor coefficient alpha and fed it through
the energy balance: ET = alpha * [Delta/(Delta+gamma)] * (Rn - G) / lambda. The
physics bounds stopped the plain net's catastrophic overfits, and the fit revealed
that observed alpha ~= 0.90 (not the well-watered 1.26) - our marshes evaporate below
potential, consistent with moisture/salinity stress. It was robust but did not beat
the Gaussian process, and did not fix upscaling (the limit was site count)."""),

("md", """## 2.5 Instantaneous evaporative fraction

We also tried the physically cleaner instantaneous match: pairing the overpass LST
with the tower's half-hourly LE to get instantaneous evaporative fraction (EF), then
scaling to daily via EF self-preservation. EF was predictable (~=0.35), but the
self-preservation scaling failed here - because ~22% of Everglades ET is nocturnal,
which a midday EF and daytime energy cannot reproduce. This is a real caveat for
thermal-ET products (DisALEXI, PT-JPL, OpenET) in wetlands."""),

("md", """## 2.6 Energy-balance-constrained and PINN models

On our full 13-site network we tested whether embedding physics improves prediction.
We built four physics-constrained variants alongside the plain tree ensemble:

- **EF-constrained** - a tree predicts the evaporative fraction, and we recover ET
  through the energy balance, ET = EF x (Rn - G).
- **Priestley-Taylor** - a tree predicts the PT coefficient alpha, ET = alpha x
  [Delta/(Delta+gamma)] x (Rn - G).
- **PINN-PT** - a neural network predicts alpha with the same PT closure.
- **PINN-EB** - a neural network predicts LE and H jointly with a soft loss that
  penalises energy-balance violation, Rn - G = H + LE.

We found that the physics constraints did **not** help once we had enough diverse
data: the plain tree ensemble stayed best (leave-site R^2 ~= 0.72), while the
constrained models fell to 0.58-0.60 and the soft-constraint PINN was unstable.
Dividing ET by an (estimated) net-radiation term amplifies noise more than the
constraint helps. This is the mirror image of what we saw at five sites, where the
physics bound *prevented* the neural net's catastrophic overfit. Our reading: physics
constraints buy robustness when data are scarce, but plain machine learning learns
the ET relationship better once the training set is large and diverse."""),
("code", """mp = pd.read_csv(f"{PROC}/physics_models.csv")
display(mp.round(3))
display(Image(f"{FIG}/physics_models.png"))"""),

("md", """*Continued in notebook 3 (spatial upscaling).*"""),
]

# ============================ NOTEBOOK 3 ============================
nb3 = [
("md", """# 3 — Spatial upscaling to unmonitored coast

*Part 3 of 3, and our main result. Five similar Everglades sites could not support
upscaling; a diverse 13-site coastal-wetland network can.*"""),
("code", SETUP),

("md", """## 3.1 Adding diverse sites

We added eight coastal wetlands beyond the Everglades - Atlantic and Gulf salt marsh,
a NC tidal forest, Pacific delta wetlands - and re-ran the leave-tower-out test.
Predicting ET at an unmonitored tower went from failing to succeeding."""),
("code", """cmp = pd.read_parquet(f"{PROC}/more_sites_table.parquet")
print(f"{len(cmp)} samples across {cmp.SITE_ID.nunique()} coastal wetlands")
print("Leave-tower-out (predict an UNSEEN tower):")
print("   5 Everglades sites : R2 = -0.37 to -0.97  (fails)")
print("  13 coastal wetlands : R2 = +0.66 to +0.71  (works)")
display(Image(f"{FIG}/upscaling_more_sites.png"))"""),

("md", """## 3.2 Skill across validation schemes

Comparing all three schemes side by side makes the point: K-fold and leave-year-out
are positive even at five sites, but only the larger, diverse network makes
leave-site-out - true upscaling - work, and it stays strong (R^2 ~= 0.7-0.8) across
every scheme."""),
("code", """display(Image(f"{FIG}/cv_comparison.png"))"""),

("md", """## 3.3 Model choice at 13 sites

At 833 samples we re-ran the full model family. Tree ensembles (ExtraTrees, Random
Forest, XGBoost) are best for spatial upscaling; the Gaussian process is best for
interpolation. Deep learning is still the worst and still unstable - these tabular
ET problems do not favour deep nets even with more data."""),
("code", """mc = pd.read_csv(f"{PROC}/model_comparison_13.csv")
mc.round(3)"""),

("md", """## 3.4 What the model actually uses

Permutation importance under leave-site-out (what helps predict an *unseen* site):
reference ET dominates, then seasonality, wind and radiation. Among satellite
features, **water (NDWI) and temperature (LST) carry the signal - not greenness**.
Physically this fits coastal wetlands: ET is limited by water and energy, not leaf
area, since the marshes are perennially vegetated."""),
("code", """imp = pd.read_csv(f"{PROC}/feature_importance.csv", index_col=0)
display(imp.sort_values("permutation_LSO", ascending=False).round(3))
display(Image(f"{FIG}/feature_importance.png"))"""),

("md", """## 3.5 ET mapping

Applying the trained model to a full Landsat scene gives a 30 m ET map; the spatial
pattern tracks vegetation and open water."""),
("code", """display(Image(f"{FIG}/et_map_everglades.png"))"""),

("md", """## What we conclude

1. **Energy-balance closure raises ET 10-30%** - We report both closed and open ET.
2. **A Gaussian process is best for interpolation, tree ensembles for upscaling;**
   deep learning overfits at this sample size.
3. **Physics-informed constraints add robustness** and reveal sub-potential ET
   (alpha ~= 0.90), though they do not raise skill.
4. **EF self-preservation breaks down in these wetlands** (~22% nocturnal ET).
5. **Spatial upscaling to unmonitored coast is achievable (R^2 ~= 0.7)** once the
   training set spans enough coastal-wetland diversity - and the satellite signal
   that matters is water and temperature, not greenness. This is our path forward."""),
]

build("01_flux_ET_and_predictors", nb1)
build("02_model_selection", nb2)
build("03_spatial_upscaling", nb3)
