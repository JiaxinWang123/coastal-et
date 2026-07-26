"""Figure: model performance with ALL vs only SOME input features."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import r2_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/anvil/scratch/x-jwang120/coastal-et"
d = pd.read_parquet(f"{R}/data/processed/more_sites_table.parquet")
SAT = ["LAI","EVI2","SAVI","NDVI","NDWI","MNDWI","LST_K"]
MET = ["TA_ERA","VPD_ERA","SW_IN_ERA","WS_ERA","ETo_mm","DOY_sin","DOY_cos"]
FEATS = SAT + MET
SITES = sorted(d.SITE_ID.unique())
RANK = ["ETo_mm","DOY_cos","WS_ERA","SW_IN_ERA","NDWI","DOY_sin","VPD_ERA"]
VIF11 = ["LAI","NDVI","MNDWI","LST_K","TA_ERA","VPD_ERA","SW_IN_ERA","WS_ERA","ETo_mm","DOY_sin","DOY_cos"]
BIC8 = ["ETo_mm","WS_ERA","DOY_cos","NDWI","VPD_ERA","DOY_sin","TA_ERA","LST_K"]
SETS = [
    ("ETo only (1)", ["ETo_mm"], "min"),
    ("greenness (4)", ["LAI","EVI2","SAVI","NDVI"], "sat"),
    ("water + LST (3)", ["NDWI","MNDWI","LST_K"], "sat"),
    ("satellite (7)", SAT, "sat"),
    ("top-6 (importance)", RANK[:6], "reduced"),
    ("top-7 (importance)", RANK[:7], "reduced"),
    ("BIC-selected (8)", BIC8, "reduced"),
    ("meteorology (7)", MET, "met"),
    ("VIF-pruned (11)", VIF11, "reduced"),
    ("FULL (14)", FEATS, "full"),
]
def evalR2(cols, scheme):
    yt,yp=[],[]
    mk=lambda: ExtraTreesRegressor(400,min_samples_leaf=2,random_state=0,n_jobs=-1)
    if scheme=="kfold":
        for tri,tei in KFold(10,shuffle=True,random_state=0).split(d):
            m=clone(mk()).fit(d.iloc[tri][cols].values,d.iloc[tri].ET_closed_mm.values)
            yp.append(m.predict(d.iloc[tei][cols].values)); yt.append(d.iloc[tei].ET_closed_mm.values)
    else:
        for s in SITES:
            tr,te=d[d.SITE_ID!=s],d[d.SITE_ID==s]
            if len(te)<5: continue
            m=clone(mk()).fit(tr[cols].values,tr.ET_closed_mm.values)
            yp.append(m.predict(te[cols].values)); yt.append(te.ET_closed_mm.values)
    return r2_score(np.concatenate(yt),np.concatenate(yp))
rows=[]
for name,cols,cat in SETS:
    ls=evalR2(cols,"site"); kf=evalR2(cols,"kfold")
    rows.append((name,len(cols),cat,round(ls,3),round(kf,3)))
    print(f"  {name:<20} n={len(cols):<3} leave-site={ls:.3f}  kfold={kf:.3f}",flush=True)
tab=pd.DataFrame(rows,columns=["feature_set","n","cat","leave_site_R2","kfold_R2"])
tab.to_csv(f"{R}/data/processed/featureset_performance.csv",index=False)
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],"font.size":9})
t=tab.sort_values("leave_site_R2")
CATc={"min":"#e0a458","sat":"#DD8452","met":"#4C72B0","reduced":"#55A868","full":"#8172B3"}
y=np.arange(len(t)); h=0.4
fig,ax=plt.subplots(figsize=(8.6,5.4))
ax.barh(y+h/2,t.leave_site_R2.clip(lower=-0.05),h,color=[CATc[c] for c in t.cat],zorder=3,label="leave-site-out (spatial transfer)")
ax.barh(y-h/2,t.kfold_R2.clip(lower=-0.05),h,color=[CATc[c] for c in t.cat],alpha=0.4,zorder=3,label="K-fold (interpolation)")
full_ls=tab[tab.feature_set=="FULL (14)"].leave_site_R2.iloc[0]
ax.axvline(full_ls,color="#8172B3",ls="--",lw=1.2,zorder=2)
ax.text(full_ls+0.006,0.15,f"full-14 = {full_ls:.2f}",fontsize=7.6,color="#8172B3",va="bottom")
for yi,(_,r) in zip(y,t.iterrows()):
    ax.text(max(r.leave_site_R2,0)+0.012,yi+h/2,f"{r.leave_site_R2:.2f}",fontsize=7.4,va="center")
ax.set_yticks(y); ax.set_yticklabels([r.feature_set for _,r in t.iterrows()],fontsize=8.7)
ax.set_xlabel("$R^2$"); ax.set_xlim(-0.1,0.9); ax.axvline(0,color="#999",lw=0.7)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
ax.legend(frameon=False,fontsize=8,loc="lower right")
ax.set_title("Model performance: all vs fewer input features (ExtraTrees)",fontsize=11.5,fontweight="bold")
fig.tight_layout()
fig.savefig(f"{R}/figures/featureset_performance.png",dpi=200,facecolor="white",bbox_inches="tight")
print("\nwrote figures/featureset_performance.png + data/processed/featureset_performance.csv")
