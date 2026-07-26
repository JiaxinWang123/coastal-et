"""AIC/BIC-based feature selection for the LINEAR model, as a complement to VIF.

AIC/BIC are defined for maximum-likelihood (here Gaussian OLS) models, so we use them to
choose a parsimonious feature subset for a linear ET model. We do forward stepwise by AIC
and by BIC, compare to the VIF-pruned set, and report both the linear (OLS) skill and the
tree-ensemble leave-site skill on each subset.

NOTE: AIC/BIC are NOT used to rank the tree ensembles themselves (no ML likelihood / no
well-defined parameter count). Model choice among those stays on leave-site-out CV.
"""
import sys, warnings
sys.path.insert(0, "/anvil/scratch/x-jwang120/coastal-et/src")
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

R = "/anvil/scratch/x-jwang120/coastal-et"
d = pd.read_parquet(f"{R}/data/processed/more_sites_table.parquet")
SAT = ["LAI", "EVI2", "SAVI", "NDVI", "NDWI", "MNDWI", "LST_K"]
MET = ["TA_ERA", "VPD_ERA", "SW_IN_ERA", "WS_ERA", "ETo_mm", "DOY_sin", "DOY_cos"]
FE = SAT + MET
SITES = sorted(d.SITE_ID.unique())
y = d.ET_closed_mm.values
Xs = pd.DataFrame(StandardScaler().fit_transform(d[FE].values), columns=FE, index=d.index)
VIF11 = ["LAI", "NDVI", "MNDWI", "LST_K", "TA_ERA", "VPD_ERA", "SW_IN_ERA", "WS_ERA", "ETo_mm", "DOY_sin", "DOY_cos"]


def ic(cols):
    m = sm.OLS(y, sm.add_constant(Xs[cols])).fit()
    return m.aic, m.bic, m.rsquared_adj


def forward(criterion):
    remaining, selected, cur = list(FE), [], np.inf
    order = []
    while remaining:
        scored = []
        for f in remaining:
            a, b, _ = ic(selected + [f])
            scored.append((f, a if criterion == "aic" else b))
        f, s = min(scored, key=lambda t: t[1])
        if s < cur - 1e-6:
            cur = s; selected.append(f); remaining.remove(f); order.append((f, round(s, 1)))
        else:
            break
    return selected, order


def leave_site(cols, kind="tree"):
    yt, yp = [], []
    for s in SITES:
        tr, te = d[d.SITE_ID != s], d[d.SITE_ID == s]
        if len(te) < 5:
            continue
        if kind == "tree":
            m = ExtraTreesRegressor(600, max_features=1.0, min_samples_leaf=2, random_state=0, n_jobs=-1)
            m.fit(tr[cols].values, tr.ET_closed_mm.values); p = m.predict(te[cols].values)
        else:
            sc = StandardScaler().fit(tr[cols].values)
            m = LinearRegression().fit(sc.transform(tr[cols].values), tr.ET_closed_mm.values)
            p = m.predict(sc.transform(te[cols].values))
        yp.append(p); yt.append(te.ET_closed_mm.values)
    return r2_score(np.concatenate(yt), np.concatenate(yp))


print("=== AIC/BIC of candidate feature sets (OLS) ===")
print(f"{'set':<18}{'k':>3}{'AIC':>10}{'BIC':>10}{'adjR2':>8}")
sets = {"full (14)": FE, "VIF-pruned (11)": VIF11}
aic_sel, aic_order = forward("aic")
bic_sel, bic_order = forward("bic")
sets[f"AIC-stepwise ({len(aic_sel)})"] = aic_sel
sets[f"BIC-stepwise ({len(bic_sel)})"] = bic_sel
for name, cols in sets.items():
    a, b, r2 = ic(cols)
    print(f"{name:<18}{len(cols):>3}{a:>10.1f}{b:>10.1f}{r2:>8.3f}")

print("\nforward-AIC order:", aic_order)
print("AIC-selected:", aic_sel)
print("\nforward-BIC order:", bic_order)
print("BIC-selected:", bic_sel)

print("\n=== leave-site R2 on each subset (linear vs tree) ===")
print(f"{'set':<18}{'OLS':>8}{'ExtraTrees':>12}")
for name, cols in sets.items():
    print(f"{name:<18}{leave_site(cols,'lin'):>8.3f}{leave_site(cols,'tree'):>12.3f}", flush=True)

pd.DataFrame([{"set": k, "k": len(v), "AIC": round(ic(v)[0], 1), "BIC": round(ic(v)[1], 1),
               "adjR2": round(ic(v)[2], 3),
               "features": ",".join(v)} for k, v in sets.items()]).to_csv(
    f"{R}/data/processed/aic_bic_selection.csv", index=False)
print(f"\nwrote data/processed/aic_bic_selection.csv")
