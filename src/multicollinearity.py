"""Feature selection by multicollinearity.

The 14 predictors have obvious redundancy: the four greenness indices are near-duplicates
(all monotone functions of (NIR-Red)), and ETo is a Penman-Monteith combination of the met
variables. We quantify it (correlation + VIF), prune to a decorrelated subset by iterative
VIF elimination, and confirm the pruned model keeps its spatial-transfer skill.
"""
import sys
sys.path.insert(0, "/anvil/scratch/x-jwang120/coastal-et/src")
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.base import clone
from sklearn.metrics import r2_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

R = "/anvil/scratch/x-jwang120/coastal-et"
d = pd.read_parquet(f"{R}/data/processed/more_sites_table.parquet")
SAT = ["LAI", "EVI2", "SAVI", "NDVI", "NDWI", "MNDWI", "LST_K"]
MET = ["TA_ERA", "VPD_ERA", "SW_IN_ERA", "WS_ERA", "ETo_mm", "DOY_sin", "DOY_cos"]
FE = SAT + MET
SITES = sorted(d.SITE_ID.unique())
X = d[FE].dropna()


def vif_table(cols):
    Z = StandardScaler().fit_transform(X[cols].values)
    out = {}
    for j, f in enumerate(cols):
        oth = [k for k in range(len(cols)) if k != j]
        r2 = LinearRegression().fit(Z[:, oth], Z[:, j]).score(Z[:, oth], Z[:, j])
        out[f] = 1.0 / max(1e-9, 1.0 - r2)
    return pd.Series(out).sort_values(ascending=False)


def iterative_vif(cols, thresh=10.0):
    cols = list(cols)
    dropped = []
    while len(cols) > 2:
        v = vif_table(cols)
        if v.max() <= thresh:
            break
        worst = v.idxmax()
        dropped.append((worst, round(float(v.max()), 1)))
        cols.remove(worst)
    return cols, dropped


def leave_site(cols):
    yt, yp = [], []
    for s in SITES:
        tr, te = d[d.SITE_ID != s], d[d.SITE_ID == s]
        if len(te) < 5:
            continue
        m = ExtraTreesRegressor(400, min_samples_leaf=2, random_state=0, n_jobs=-1)
        m.fit(tr[cols].values, tr.ET_closed_mm.values)
        yp.append(m.predict(te[cols].values)); yt.append(te.ET_closed_mm.values)
    return r2_score(np.concatenate(yt), np.concatenate(yp))


print("=== 1. Correlation with target and worst pairwise correlations ===")
corr = d[FE + ["ET_closed_mm"]].corr()
print("|r| with ET (top):")
print(corr["ET_closed_mm"].drop("ET_closed_mm").abs().sort_values(ascending=False).round(2).to_string())
c = corr.loc[FE, FE].abs()
pairs = [(FE[i], FE[j], c.iloc[i, j]) for i in range(len(FE)) for j in range(i + 1, len(FE))]
pairs.sort(key=lambda t: -t[2])
print("\nmost collinear pairs (|r|):")
for a, b, r in pairs[:8]:
    print(f"  {a:<10} {b:<10} {r:.2f}")

print("\n=== 2. VIF of the full 14-feature set ===")
v = vif_table(FE)
for f, val in v.items():
    flag = "  <-- severe" if val > 10 else ("  <- high" if val > 5 else "")
    print(f"  {f:<11} VIF={val:8.1f}{flag}")

print("\n=== 3. Iterative VIF elimination (drop worst until all VIF <= 10) ===")
keep_vif, dropped = iterative_vif(FE, 10.0)
for f, val in dropped:
    print(f"  dropped {f:<11} (VIF was {val})")
print(f"\nVIF-selected ({len(keep_vif)}): {keep_vif}")
print("final VIFs:")
print(vif_table(keep_vif).round(2).to_string())

print("\n=== 4. Correlation-cluster selection (|r|>0.8 groups; keep best-with-ET each) ===")
dist = squareform(1 - c.values, checks=False)
Z = linkage(dist, method="average")
labels = fcluster(Z, t=0.2, criterion="distance")   # 1-|r| < 0.2  => |r| > 0.8
tgt = corr["ET_closed_mm"].abs()
keep_clu = []
for lab in sorted(set(labels)):
    grp = [FE[i] for i in range(len(FE)) if labels[i] == lab]
    rep = max(grp, key=lambda f: tgt[f])
    keep_clu.append(rep)
    if len(grp) > 1:
        print(f"  cluster {lab}: {grp}  -> keep {rep}")
print(f"cluster-selected ({len(keep_clu)}): {sorted(keep_clu)}")

print("\n=== 5. Spatial-transfer skill: does pruning hurt? (leave-site R2) ===")
for name, cols in [("FULL (14)", FE), (f"VIF-selected ({len(keep_vif)})", keep_vif),
                   (f"cluster-selected ({len(keep_clu)})", keep_clu)]:
    print(f"  {name:<22} leave-site R2 = {leave_site(cols):.3f}")

# ---- figure: correlation heatmap + full-set VIF ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
im = ax1.imshow(c.values, cmap="RdBu_r", vmin=-0, vmax=1)
ax1.set_xticks(range(len(FE))); ax1.set_xticklabels(FE, rotation=90, fontsize=6.5)
ax1.set_yticks(range(len(FE))); ax1.set_yticklabels(FE, fontsize=6.5)
ax1.set_title("|correlation| among predictors", fontsize=9.5, fontweight="bold")
fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.02)
vv = vif_table(FE)
col = ["#c0392b" if x > 10 else ("#e67e22" if x > 5 else "#4C72B0") for x in vv.values]
ax2.barh(range(len(vv)), np.clip(vv.values, 0, 60), color=col, zorder=3)
ax2.set_yticks(range(len(vv))); ax2.set_yticklabels(vv.index, fontsize=7)
ax2.axvline(10, color="#c0392b", lw=0.8, ls="--"); ax2.axvline(5, color="#e67e22", lw=0.8, ls=":")
ax2.set_xlabel("VIF (clipped at 60)"); ax2.invert_yaxis()
for sp in ("top", "right"): ax2.spines[sp].set_visible(False)
ax2.set_title("Variance inflation factor", fontsize=9.5, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{R}/figures/multicollinearity.png", dpi=200, bbox_inches="tight", facecolor="white")
pd.DataFrame({"VIF_full": vif_table(FE)}).to_csv(f"{R}/data/processed/multicollinearity_vif.csv")
print(f"\nwrote figures/multicollinearity.png and multicollinearity_vif.csv")
print(f"VIF-selected set: {keep_vif}")
