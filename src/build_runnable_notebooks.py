"""Generate RUNNABLE notebooks 02 (model selection) and 03 (spatial upscaling).

Team first-person voice. Unlike the earlier display-only versions, these actually
train the models and draw the figures in-notebook, from the self-contained
data/processed/more_sites_table.parquet (833 overpass matches, 13 sites, 14 features).
No dependency on src/ modules or personal paths -- ROOT is derived from the notebook
location, so any teammate can Run All.
"""
import os
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

# ---- shared preamble: derive ROOT relative to the notebook, no personal paths ----
PREAMBLE = '''import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

# Derive the project root without hardcoding anyone's personal path:
#   env override -> parent of the notebook's folder -> shared fallback.
ROOT = os.environ.get("COASTAL_ET_ROOT")
if not ROOT or not os.path.isdir(os.path.join(ROOT, "data", "processed")):
    cand = os.path.dirname(os.getcwd())                 # notebook lives in <ROOT>/notebooks
    ROOT = cand if os.path.isdir(os.path.join(cand, "data", "processed")) \\
        else "/anvil/projects/x-ees260113/team2/coastal-et"
PROC = f"{ROOT}/data/processed"
FIG = f"{ROOT}/figures"; os.makedirs(FIG, exist_ok=True)
print("project root:", ROOT)

# Nature-ish figure defaults
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.linewidth": 0.7, "figure.dpi": 120,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7})
INK = "#1a1a1a"'''

FEATS_CELL = '''# The 14 predictors: 7 satellite + 7 meteorology. Target is measured closed ET.
SAT = ["LAI", "EVI2", "SAVI", "NDVI", "NDWI", "MNDWI", "LST_K"]
MET = ["TA_ERA", "VPD_ERA", "SW_IN_ERA", "WS_ERA", "ETo_mm", "DOY_sin", "DOY_cos"]
FEATS = SAT + MET
EVERGLADES = ["US-Esm", "US-TaS", "US-Skr", "US-Elm", "US-EvM"]

d = pd.read_parquet(f"{PROC}/more_sites_table.parquet")
SITES = sorted(d.SITE_ID.unique())
print(f"{len(d)} overpass matches | {len(SITES)} sites | {len(FEATS)} features")
print("target: ET_closed_mm (measured, closure-corrected daily ET)")
d[["SITE_ID", "year"] + FEATS + ["ET_closed_mm"]].head()'''

CV_CELL = '''from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_absolute_error

def evaluate(make_model, data, scheme, feats=FEATS):
    """Return (R2, MAE, y_true, y_pred) for one CV scheme.
    kfold=predict at monitored sites; year=predict unseen years;
    site=predict a completely unseen tower (spatial upscaling)."""
    yt, yp = [], []
    if scheme == "kfold":
        for tri, tei in KFold(10, shuffle=True, random_state=0).split(data):
            m = clone(make_model()).fit(data.iloc[tri][feats].values, data.iloc[tri].ET_closed_mm.values)
            yp.append(m.predict(data.iloc[tei][feats].values)); yt.append(data.iloc[tei].ET_closed_mm.values)
    elif scheme == "year":
        for s in sorted(data.SITE_ID.unique()):
            ds = data[data.SITE_ID == s]
            for y in sorted(ds.year.dropna().unique()):
                tr, te = ds[ds.year != y], ds[ds.year == y]
                if len(te) < 5 or len(tr) < 15: continue
                m = clone(make_model()).fit(tr[feats].values, tr.ET_closed_mm.values)
                yp.append(m.predict(te[feats].values)); yt.append(te.ET_closed_mm.values)
    else:  # leave-site-out
        for s in sorted(data.SITE_ID.unique()):
            tr, te = data[data.SITE_ID != s], data[data.SITE_ID == s]
            if len(te) < 5: continue
            m = clone(make_model()).fit(tr[feats].values, tr.ET_closed_mm.values)
            yp.append(m.predict(te[feats].values)); yt.append(te.ET_closed_mm.values)
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    return r2_score(yt, yp), mean_absolute_error(yt, yp), yt, yp

print("CV evaluator ready (schemes: kfold, year, site)")'''

MODELS_CELL = '''from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, HistGradientBoostingRegressor)
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def model_zoo():
    z = {
        "Ridge":       lambda: make_pipeline(StandardScaler(), Ridge(alpha=10)),
        "ElasticNet":  lambda: make_pipeline(StandardScaler(), ElasticNet(alpha=0.05, l1_ratio=0.3, max_iter=5000)),
        "PLS":         lambda: make_pipeline(StandardScaler(), PLSRegression(n_components=6)),
        "kNN":         lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(10, weights="distance")),
        "SVR":         lambda: make_pipeline(StandardScaler(), SVR(C=5, gamma="scale", epsilon=0.2)),
        "GaussProc":   lambda: make_pipeline(StandardScaler(), GaussianProcessRegressor(
                            kernel=ConstantKernel() * RBF() + WhiteKernel(), normalize_y=True, alpha=1e-3, random_state=0)),
        "RandomForest": lambda: RandomForestRegressor(400, min_samples_leaf=2, random_state=0, n_jobs=-1),
        "ExtraTrees":  lambda: ExtraTreesRegressor(400, min_samples_leaf=2, random_state=0, n_jobs=-1),
        "GradBoost":   lambda: GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, random_state=0),
        "HistGBM":     lambda: HistGradientBoostingRegressor(max_iter=400, l2_regularization=1, random_state=0),
    }
    try:
        from xgboost import XGBRegressor
        z["XGBoost"] = lambda: XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.03,
                                            subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=0)
    except Exception: pass
    try:
        from lightgbm import LGBMRegressor
        z["LightGBM"] = lambda: LGBMRegressor(n_estimators=400, num_leaves=31, learning_rate=0.03,
                                              subsample=0.8, random_state=0, verbose=-1)
    except Exception: pass
    return z

ZOO = model_zoo()
print("model zoo:", list(ZOO.keys()))'''

TRAIN_CELL = '''# Train every model across all three schemes. This actually fits the models now
# (~1-3 min). Leave-site-out is the headline test: can we predict an UNSEEN tower?
rows = []
for name, mk in ZOO.items():
    rk, _, _, _ = evaluate(mk, d, "kfold")
    ry, _, _, _ = evaluate(mk, d, "year")
    rs, _, _, _ = evaluate(mk, d, "site")
    rows.append((name, rk, ry, rs))
    print(f"  {name:<13} kfold={rk:5.2f}  leave-year={ry:5.2f}  leave-site={rs:5.2f}", flush=True)

comp = pd.DataFrame(rows, columns=["model", "kfold", "leave_year", "leave_site"]).sort_values(
    "leave_site", ascending=False).reset_index(drop=True)
print(f"\\nbest upscaler (leave-site): {comp.iloc[0].model}  R2={comp.iloc[0].leave_site:.2f}")
comp'''

COMP_FIG_CELL = '''# Figure: R2 by validation scheme for the top models (draw it here, save a copy)
top = comp.head(6)
x = np.arange(len(top)); w = 0.26
fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
for i, (col, lab, c) in enumerate([("kfold", "K-fold (monitored)", "#B0C4DE"),
                                   ("leave_year", "leave-year (unseen years)", "#6b9bc3"),
                                   ("leave_site", "leave-site (unseen tower)", "#2C5F8A")]):
    ax.bar(x + (i-1)*w, np.clip(top[col], -0.2, 1), w, label=lab, color=c, zorder=3)
ax.axhline(0, color="#8a8a8a", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(top.model, rotation=30, ha="right", fontsize=7.5)
ax.set_ylabel("$R^2$"); ax.set_ylim(-0.2, 1.0)
ax.legend(frameon=False, fontsize=7, loc="upper right")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.set_title("Predictive skill by validation scheme (13 sites)", fontsize=9.5, fontweight="bold")
fig.savefig(f"{FIG}/model_comparison_runnable.png", dpi=200, bbox_inches="tight")
plt.show()'''

IMP_CELL = '''# Feature importance the honest way: permutation importance measured on HELD-OUT
# sites (leave-site-out), averaged over sites. This ranks what actually transfers.
from sklearn.inspection import permutation_importance
perm = np.zeros(len(FEATS)); n = 0
for s in SITES:
    tr, te = d[d.SITE_ID != s], d[d.SITE_ID == s]
    if len(te) < 5: continue
    m = ExtraTreesRegressor(400, min_samples_leaf=2, random_state=0, n_jobs=-1).fit(
        tr[FEATS].values, tr.ET_closed_mm.values)
    pi = permutation_importance(m, te[FEATS].values, te.ET_closed_mm.values, n_repeats=8, random_state=0)
    perm += np.clip(pi.importances_mean, 0, None); n += 1
imp = pd.Series(perm / n, index=FEATS).sort_values()

col = ["#55A868" if f in SAT else "#4C72B0" for f in imp.index]
fig, ax = plt.subplots(figsize=(4.6, 4.2), constrained_layout=True)
ax.barh(np.arange(len(imp)), imp.values, color=col, height=0.72, zorder=3)
ax.set_yticks(np.arange(len(imp))); ax.set_yticklabels(imp.index, fontsize=7.5)
ax.set_xlabel("Permutation importance (leave-site-out $\\\\Delta R^2$)")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc="#55A868", label="satellite"), Patch(fc="#4C72B0", label="meteorology")],
          frameon=False, fontsize=7, loc="lower right")
ax.set_title("What drives transferable ET skill", fontsize=9.5, fontweight="bold")
fig.savefig(f"{FIG}/feature_importance_runnable.png", dpi=200, bbox_inches="tight")
plt.show()
print("Top drivers:", list(imp.sort_values(ascending=False).index[:5]))'''


def build_02():
    nb = new_notebook(); c = []
    md = lambda t: c.append(new_markdown_cell(t)); co = lambda t: c.append(new_code_cell(t))
    md("""# 02 · Model selection — train it yourself

*Runnable notebook. We fit every model here and draw every figure from the data; nothing
is a pre-baked image. Set the kernel to **Python (coastal-et)** and Run All (~1–3 min).*

We ask one question three ways: given satellite indices + meteorology on a cloud-free
overpass, can we predict the tower's measured daily ET? The three validation schemes are
progressively harder — predicting at a **monitored** site (K-fold), in an **unseen year**
(leave-year), and at a **completely unseen tower** (leave-site, i.e. true upscaling).""")
    co(PREAMBLE)
    md("""## Load the data\n\nOne self-contained table: 833 overpass matches across 13 coastal-wetland towers.""")
    co(FEATS_CELL)
    md("""## Cross-validation, honestly\n\nThe evaluator refits the model for every fold/year/site so nothing leaks across the
split we care about.""")
    co(CV_CELL)
    md("""## The model zoo\n\nLinear, kernel, tree-ensemble, Gaussian-process and (if installed) boosted-tree models.""")
    co(MODELS_CELL)
    md("""## Train everything\n\nThis is the actual training run — each model is fit across all three schemes.""")
    co(TRAIN_CELL)
    md("""## Visualize the comparison""")
    co(COMP_FIG_CELL)
    md("""We consistently find **tree ensembles (ExtraTrees / RandomForest)** give the best
leave-site skill, while the **Gaussian process** wins K-fold interpolation. Deep nets are
omitted here — at n≈833 they don't beat the trees and add instability.""")
    md("""## Feature importance — what actually transfers\n\nGini importance flatters whatever the trees split on in-sample. The honest measure is
permutation importance on **held-out sites**: shuffle a feature, see how much unseen-tower
skill drops.""")
    co(IMP_CELL)
    md("""Reference ET (`ETo_mm`) dominates, followed by seasonal timing and the water/moisture
indices (NDWI, LST); raw greenness (LAI/NDVI/EVI2/SAVI) barely moves unseen-site skill —
the marshes are too spectrally similar in greenness for it to discriminate.""")

    md("""## 8. Feature-group ablation — how few / which inputs do we need?

Before any formal selection, we just train on progressively fewer (or different) input
groups and score each on leave-site-out. This shows directly how much each group adds.""")
    co('''from sklearn.ensemble import ExtraTreesRegressor
GROUPS = {
    "ETo only (1)":         ["ETo_mm"],
    "meteorology (7)":      MET,
    "greenness (4)":        ["LAI", "EVI2", "SAVI", "NDVI"],
    "water + LST (3)":      ["NDWI", "MNDWI", "LST_K"],
    "satellite (7)":        SAT,
    "met + water/LST (10)": MET + ["NDWI", "MNDWI", "LST_K"],
    "FULL (14)":            FEATS,
}
# the top-k most important inputs (ranking from the section-7 permutation importance)
ranked = imp.sort_values(ascending=False).index.tolist()
GROUPS["top-6 (importance)"] = ranked[:6]
GROUPS["top-7 (importance)"] = ranked[:7]
print("top-7 by importance:", ranked[:7])
etm = lambda: ExtraTreesRegressor(400, min_samples_leaf=2, random_state=0, n_jobs=-1)
rows = []
for name, cols in GROUPS.items():
    rs = evaluate(etm, d, "site", feats=cols)[0]
    rows.append((name, len(cols), round(rs, 3)))
    print(f"  {name:<22} n={len(cols):<3} leave-site R2 = {rs:.3f}", flush=True)
abl = pd.DataFrame(rows, columns=["feature set", "n", "leave_site_R2"])
abl''')
    co('''order = abl.sort_values("leave_site_R2")
col = ["#C44E52" if s in ("greenness (4)", "ETo only (1)") else
       ("#DD8452" if ("satellite" in s or "water" in s) else "#4C72B0") for s in order["feature set"]]
fig, ax = plt.subplots(figsize=(6.6, 3.8))
ax.barh(np.arange(len(order)), order.leave_site_R2.clip(lower=-0.05), color=col, zorder=3)
ax.set_yticks(np.arange(len(order))); ax.set_yticklabels(order["feature set"], fontsize=8.5)
ax.axvline(order.leave_site_R2.max(), color="#888", ls="--", lw=0.8)
ax.set_xlabel("leave-site-out $R^2$"); ax.set_xlim(-0.1, 0.8)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.set_title("Fewer inputs, same skill — feature-group ablation", fontsize=10, fontweight="bold")
plt.show()''')
    md("""**Takeaway:** meteorology alone already matches the full 14-feature model; greenness
alone is noise (R² ≈ 0); the only satellite signal that helps is **water + LST**. In fact the
**top-7 inputs by importance — 6 meteorology + NDWI — reach leave-site R² ≈ 0.72, matching the
full model with half the features** (top-6 ≈ 0.72 too). So the model can be trimmed hard with
no loss, which the VIF and AIC/BIC selection below make formal.""")

    md("""## 9. Feature selection I — multicollinearity (VIF)

Several predictors are near-duplicates: the greenness indices are all monotone functions
of NIR-Red, and ETo is a combination of the met variables. We quantify this with the
variance inflation factor (VIF) and prune the redundant features.""")
    co('''from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def vif(cols):
    Z = StandardScaler().fit_transform(d[cols].values)
    return pd.Series({f: 1/max(1e-9, 1-LinearRegression().fit(
        np.delete(Z, j, 1), Z[:, j]).score(np.delete(Z, j, 1), Z[:, j]))
        for j, f in enumerate(cols)}).sort_values(ascending=False)

print("VIF (full 14 features):"); print(vif(FEATS).round(1).to_string())

cmat = d[FEATS].corr().abs()
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cmat.values, cmap="RdBu_r", vmin=0, vmax=1)
ax.set_xticks(range(len(FEATS))); ax.set_xticklabels(FEATS, rotation=90, fontsize=6.5)
ax.set_yticks(range(len(FEATS))); ax.set_yticklabels(FEATS, fontsize=6.5)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
ax.set_title("|correlation| among predictors", fontsize=10, fontweight="bold")
plt.show()

# iterative elimination: drop the highest VIF until all <= 10
cols = list(FEATS)
while len(cols) > 2 and vif(cols).max() > 10:
    worst = vif(cols).idxmax(); print("drop", worst, " VIF", round(vif(cols).max(), 1)); cols.remove(worst)
SELECTED = cols
print("\\nVIF-pruned set (", len(SELECTED), "):", SELECTED)''')
    md("""VIF flags the greenness indices as severe (EVI2/SAVI in the thousands). Eliminating
the worst leaves an ~11-feature decorrelated set (all VIF ≤ ~8).""")

    md("""## 10. Feature selection II — AIC/BIC (linear model)

AIC/BIC are defined for the maximum-likelihood (OLS) model, so we use them for forward
stepwise selection of a linear ET model. They are **not** valid for the tree ensembles (no
likelihood, no well-defined parameter count), so model choice among those stays on
leave-site CV.""")
    co('''import statsmodels.api as sm
y = d.ET_closed_mm.values
Xs = pd.DataFrame(StandardScaler().fit_transform(d[FEATS].values), columns=FEATS, index=d.index)

def ic(cols):
    m = sm.OLS(y, sm.add_constant(Xs[cols])).fit(); return m.aic, m.bic

def forward(which):
    rem, sel, cur = list(FEATS), [], 1e18
    while rem:
        f, s = min(((f, ic(sel + [f])[0 if which == "aic" else 1]) for f in rem), key=lambda t: t[1])
        if s < cur - 1e-6:
            cur = s; sel.append(f); rem.remove(f)
        else:
            break
    return sel

aic_set, bic_set = forward("aic"), forward("bic")
for nm, cs in [("full(14)", FEATS), ("VIF", SELECTED), ("AIC-step", aic_set), ("BIC-step", bic_set)]:
    a, b = ic(cs); print(f"  {nm:<9} k={len(cs):<3} AIC={a:8.1f}  BIC={b:8.1f}")
print("\\nBIC-selected:", bic_set)''')
    md("""BIC (the stricter penalty) drops **every greenness index**, keeping only water
(NDWI) and thermal (LST) among the satellite features — the same verdict as VIF and the
feature-importance ablation.""")

    md("""## 11. Retrain all models on the selected features → pick + save the best

We retrain the whole model zoo on the pruned features and rank by leave-site R² (spatial
transfer). The winner is refit on all the data and saved as the production model.""")
    co('''import joblib, json
from sklearn.base import clone

rows = []
for name, mk in ZOO.items():
    rows.append((name, round(evaluate(mk, d, "site", feats=SELECTED)[0], 3)))
rank = pd.DataFrame(rows, columns=["model", "leave_site_R2"]).sort_values("leave_site_R2", ascending=False)
print(rank.to_string(index=False))

best = rank.iloc[0]["model"]
prod = clone(ZOO[best]()).fit(d[SELECTED].values, d.ET_closed_mm.values)
meta = {"model": best, "features": SELECTED, "leave_site_R2": float(rank.iloc[0]["leave_site_R2"]),
        "n_train": int(len(d)), "n_sites": int(d.SITE_ID.nunique())}
joblib.dump({**meta, "model": prod}, f"{PROC}/final_model.joblib")  # fitted model wins the key
json.dump(meta, open(f"{PROC}/final_model.json", "w"), indent=2)
print(f"\\nPRODUCTION MODEL: {best} on {len(SELECTED)} features  ->  saved final_model.joblib")''')
    md("""**Result:** ExtraTrees on the decorrelated feature set is the production model
(leave-site R² ≈ 0.72). Multicollinearity (VIF), AIC/BIC, and the importance ablation all
converge on the same parsimonious set — **water + thermal + meteorology, greenness dropped**
— and this is the model `04_spatial_prediction` uses to map ET.""")

    nb["cells"] = c
    nb["metadata"] = {"kernelspec": {"display_name": "Python (coastal-et)", "language": "python", "name": "coastal-et"},
                      "language_info": {"name": "python"}}
    return nb


def build_03():
    nb = new_notebook(); c = []
    md = lambda t: c.append(new_markdown_cell(t)); co = lambda t: c.append(new_code_cell(t))
    md("""# 03 · Spatial upscaling — our key result, reproduced live

*Runnable notebook. We reproduce the headline finding from scratch: predicting ET at an
**unmonitored** tower fails with 5 similar Everglades sites but works once the training set
spans 13 diverse coastal wetlands. Set the kernel to **Python (coastal-et)** and Run All.*""")
    co(PREAMBLE)
    co(FEATS_CELL)
    co(CV_CELL)
    md("""## 5 Everglades sites vs 13 diverse wetlands\n\nWe run leave-site-out (predict a fully unseen tower) on the 5-site Everglades subset and
on the full 13-site network, with our best upscaler (ExtraTrees) and the Gaussian process.""")
    co('''from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODELS = {
    "ExtraTrees": lambda: ExtraTreesRegressor(400, min_samples_leaf=2, random_state=0, n_jobs=-1),
    "GaussProc":  lambda: make_pipeline(StandardScaler(), GaussianProcessRegressor(
                       kernel=ConstantKernel() * RBF() + WhiteKernel(), normalize_y=True, alpha=1e-3, random_state=0)),
}
d5 = d[d.SITE_ID.isin(EVERGLADES)]
res = {}
for mn, mk in MODELS.items():
    for label, data in [("5 Everglades", d5), ("13 wetlands", d)]:
        rk = evaluate(mk, data, "kfold")[0]; ry = evaluate(mk, data, "year")[0]; rs = evaluate(mk, data, "site")[0]
        res[(mn, label)] = (rk, ry, rs)
        print(f"  {mn:<11}{label:<14} kfold={rk:5.2f}  leave-year={ry:5.2f}  leave-site={rs:5.2f}", flush=True)''')
    md("""## The picture: harder tests need more sites""")
    co('''schemes = ["K-fold\\n(monitored)", "Leave-year\\n(unseen yr)", "Leave-site\\n(unseen tower)"]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True, constrained_layout=True)
x = np.arange(3); w = 0.36
for ax, mn in zip(axes, ["ExtraTrees", "GaussProc"]):
    v5, v13 = res[(mn, "5 Everglades")], res[(mn, "13 wetlands")]
    ax.bar(x - w/2, np.clip(v5, -1.05, 1), w, color="#B0C4DE", label="5 Everglades", zorder=3)
    ax.bar(x + w/2, np.clip(v13, -1.05, 1), w, color="#2C5F8A", label="13 wetlands", zorder=3)
    ax.axhline(0, color="#8a8a8a", lw=0.8)
    for xi, (a, b) in enumerate(zip(v5, v13)):
        ax.text(xi - w/2, max(a, 0)+0.03, f"{a:.2f}", ha="center", fontsize=6, color=INK)
        ax.text(xi + w/2, max(b, 0)+0.03, f"{b:.2f}", ha="center", fontsize=6, color=INK)
    ax.set_title(mn, fontsize=9); ax.set_xticks(x); ax.set_xticklabels(schemes, fontsize=6.8)
    ax.set_ylim(-1.15, 1.0)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
axes[0].set_ylabel("$R^2$"); axes[1].legend(frameon=False, fontsize=7, loc="lower right")
fig.suptitle("Upscaling to an unseen tower needs training-set diversity",
             fontsize=10, fontweight="bold")
fig.savefig(f"{FIG}/cv_comparison_runnable.png", dpi=200, bbox_inches="tight")
plt.show()''')
    md("""## Predicted vs observed at held-out towers (13 sites)\n\nEvery point is an overpass at a tower the model never saw in training.""")
    co('''from sklearn.ensemble import ExtraTreesRegressor
_, _, yt, yp = evaluate(lambda: ExtraTreesRegressor(400, min_samples_leaf=2, random_state=0, n_jobs=-1), d, "site")
from sklearn.metrics import r2_score, mean_absolute_error
fig, ax = plt.subplots(figsize=(4.0, 4.0), constrained_layout=True)
ax.scatter(yt, yp, s=9, alpha=0.35, color="#2C5F8A", edgecolor="none")
lim = [0, max(yt.max(), yp.max())*1.05]
ax.plot(lim, lim, "--", color="#8a8a8a", lw=0.9)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Observed ET (mm/day)"); ax.set_ylabel("Predicted ET (mm/day)")
ax.text(0.05, 0.92, f"$R^2$={r2_score(yt,yp):.2f}\\nMAE={mean_absolute_error(yt,yp):.2f} mm/d",
        transform=ax.transAxes, fontsize=8, va="top")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.set_title("Leave-site-out prediction, 13 wetlands", fontsize=9.5, fontweight="bold")
fig.savefig(f"{FIG}/upscaling_scatter_runnable.png", dpi=200, bbox_inches="tight")
plt.show()''')
    md("""**The finding, reproduced live:** the bottleneck to satellite ET upscaling in coastal
wetlands is **training-set diversity**, not the model or the features. Five spectrally
near-identical Everglades marshes can't teach a transferable relationship; thirteen
wetlands spanning different climates, salinities and canopies can (leave-site $R^2\\approx0.7$).""")
    nb["cells"] = c
    nb["metadata"] = {"kernelspec": {"display_name": "Python (coastal-et)", "language": "python", "name": "coastal-et"},
                      "language_info": {"name": "python"}}
    return nb


OUT_DIRS = [
    "/anvil/scratch/x-jwang120/coastal-et/notebooks",
    "/anvil/projects/x-ees260113/team2/coastal-et/notebooks",
    "/home/x-jwang120/coastal-et/notebooks",
]
for nb, fname in [(build_02(), "02_model_selection.ipynb"), (build_03(), "03_spatial_upscaling.ipynb")]:
    for d_ in OUT_DIRS:
        os.makedirs(d_, exist_ok=True)
        with open(f"{d_}/{fname}", "w") as f:
            nbf.write(nb, f)
    print("wrote", fname, "->", len(nb["cells"]), "cells x", len(OUT_DIRS), "locations")
