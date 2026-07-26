"""Workflow diagram: coastal-wetland ET, data -> prediction, with implementation + impacts."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

R = "/anvil/scratch/x-jwang120/coastal-et"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"]})

CAT = {"input": "#4C72B0", "process": "#55A868", "data": "#8172B3", "model": "#DD8452", "output": "#C44E52"}
# (title, detail, category, implementation-note)
STAGES = [
    ("1 · DATA DOWNLOAD", "AmeriFlux/FLUXNET  ·  Landsat C2 + Sentinel-2 (Planetary Computer)  ·  ERA5 + gridMET", "input"),
    ("2 · FLUX ET  (target)", "Energy-balance closure (Bowen / ONEFlux LE_CORR) → daily ET, ≥80% coverage (+10–30%)", "process"),
    ("3 · PREDICTORS", "7 satellite indices per-pixel (S2 fused to Landsat) + 7 met (ETo, cyclic DOY)", "process"),
    ("4 · ANALYSIS TABLE", "833 cloud-free overpass site-days  ·  13 diverse coastal wetlands  ·  14 features", "data"),
    ("5 · MODEL SELECTION", "12 models × 3 CV schemes; leave-site-out is the headline  →  ExtraTrees wins", "model"),
    ("6 · FEATURE SELECTION", "VIF + AIC/BIC + ablation → 11 features (drop greenness); footprint tested — no gain", "model"),
    ("7 · FINAL MODEL", "ExtraTrees, 11 features  ·  leave-site-out R² ≈ 0.72, MAE ≈ 0.8 mm/day", "model"),
    ("8 · SPATIAL PREDICTION", "30 m ET maps over NERR reserves — per-pixel, open water masked, clipped to boundary", "output"),
]

IMPL = [
    "Anvil HPC (Slurm); conda env Python 3.11, shared team kernel",
    "Idempotent download/preprocess (skip-if-exists, --force)",
    "Kljun 2015 footprint (fluxfootprints) for all 13 sites",
    "5 runnable notebooks 00–04; production model saved (joblib)",
    "gridMET via OPeNDAP; STAC + stackstac for imagery",
]
IMPACT = [
    "Upscaling works (R²≈0.72) only with a DIVERSE 13-site network —",
    "   5 spectrally-similar Everglades sites fail (R²<0).",
    "Bottleneck = training-set DIVERSITY, not model/features/physics.",
    "Daily wetland ET is DEMAND-limited: met carries temporal skill;",
    "   satellite's role is 30 m SPATIAL disaggregation.",
    "Footprint weighting ≈ window (support-invariant) — 30 m justified.",
    "Delivered 30 m ET maps for 7 SE-US coastal reserves.",
]

fig = plt.figure(figsize=(15, 9.6)); fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

ax.text(2.5, 97, "Coastal-wetland evapotranspiration — workflow from data to prediction",
        fontsize=18, fontweight="bold", color="#111")
ax.text(2.5, 93.5, "Flux-tower ET → satellite/met predictors → machine-learning upscaling → 30 m ET maps",
        fontsize=11.5, color="#555")

# ---- left: pipeline of stage boxes ----
x0, w = 2.5, 60
n = len(STAGES); top = 88; bot = 5; gap = 1.6
h = (top - bot - gap * (n - 1)) / n
for i, (title, detail, cat) in enumerate(STAGES):
    y = top - i * (h + gap) - h
    c = CAT[cat]
    ax.add_patch(FancyBboxPatch((x0, y), w, h, boxstyle="round,pad=0.15,rounding_size=1.2",
                                fc=c, ec="none", alpha=0.16, zorder=2))
    ax.add_patch(FancyBboxPatch((x0, y), 0.9, h, boxstyle="round,pad=0,rounding_size=0.3",
                                fc=c, ec="none", zorder=3))
    ax.text(x0 + 2.2, y + h * 0.66, title, fontsize=12.5, fontweight="bold", color="#1a1a1a", va="center")
    ax.text(x0 + 2.2, y + h * 0.26, detail, fontsize=9.3, color="#444", va="center")
    if i < n - 1:
        ax.add_patch(FancyArrowPatch((x0 + w / 2, y - 0.1), (x0 + w / 2, y - gap + 0.1),
                                     arrowstyle="-|>", mutation_scale=16, color="#999", lw=1.6, zorder=4))

# ---- right: implementation + impacts panels ----
rx, rw = 67, 30.5
# implementation
iy, ih = 60, 27
ax.add_patch(FancyBboxPatch((rx, iy), rw, ih, boxstyle="round,pad=0.3,rounding_size=1.2",
                            fc="#f4f6f8", ec="#d0d7de", lw=1.0, zorder=2))
ax.text(rx + 1.6, iy + ih - 2.4, "IMPLEMENTATION", fontsize=12, fontweight="bold", color="#24486b")
for j, t in enumerate(IMPL):
    ax.text(rx + 1.6, iy + ih - 6.2 - j * 3.9, "• " + t, fontsize=9.2, color="#333", va="top", wrap=True)

# impacts
ky, kh = 5, 52
ax.add_patch(FancyBboxPatch((rx, ky), rw, kh, boxstyle="round,pad=0.3,rounding_size=1.2",
                            fc="#fbf3f2", ec="#e6c9c6", lw=1.0, zorder=2))
ax.text(rx + 1.6, ky + kh - 2.6, "IMPACTS & KEY FINDINGS", fontsize=12, fontweight="bold", color="#8f342f")
for j, t in enumerate(IMPACT):
    bold = not t.startswith("   ")
    ax.text(rx + 1.6, ky + kh - 7 - j * 5.3, ("• " if bold else "   ") + t.strip(),
            fontsize=9.5 if bold else 9.0, color="#2a2a2a" if bold else "#555",
            fontweight="bold" if bold else "normal", va="top")
# headline number callout
ax.add_patch(FancyBboxPatch((rx + 1.6, ky + 2.2, ), rw - 3.2, 8.4, boxstyle="round,pad=0.2,rounding_size=0.8",
                            fc="#C44E52", ec="none", alpha=0.9, zorder=3))
ax.text(rx + rw / 2, ky + 8.0, "leave-site-out  R² ≈ 0.72", fontsize=13.5, fontweight="bold",
        color="white", ha="center", va="center", zorder=4)
ax.text(rx + rw / 2, ky + 4.3, "predict ET at an unmonitored coastal wetland", fontsize=8.6,
        color="white", ha="center", va="center", zorder=4)

fig.savefig(f"{R}/figures/workflow_diagram.png", dpi=200, facecolor="white", bbox_inches="tight")
print("wrote figures/workflow_diagram.png")
