"""Spatial upscaling test: train on 3 long-record sites, predict UNSEEN sites.
TRAIN Esm/TaS/Skr (2018-2021)   TEST Elm/EvM (all their data)
Uses the improved feature pipeline (merged met, corrected NDVI, ECOSTRESS)."""
import os, sys; sys.path.insert(0,"/anvil/scratch/x-jwang120/coastal-et/src")
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import gapfill_model as G

TRAIN=["US-Esm","US-TaS","US-Skr"]; TEST=["US-Elm","US-EvM"]
FEATS=G.FEATS
frames={s:G.build_site(s) for s in TRAIN+TEST}
tr=pd.concat([frames[s][(frames[s].index>="2018-01-01")&(frames[s].index<="2021-12-31")] for s in TRAIN])
te=pd.concat([frames[s] for s in TEST])
tr=tr.dropna(subset=["ET_closed_mm"]+FEATS); te=te.dropna(subset=["ET_closed_mm"]+FEATS)
Xtr,ytr=tr[FEATS].values,tr.ET_closed_mm.values
print(f"TRAIN {TRAIN} 2018-2021: {len(tr)} rows (mean ET {ytr.mean():.2f})")
print(f"TEST  {TEST}: {len(te)} rows (mean ET {te.ET_closed_mm.mean():.2f})")
print(f"  -> target-mean gap: {te.ET_closed_mm.mean()-ytr.mean():+.2f} mm/d\n")
def sc(y,p): return dict(R2=r2_score(y,p),MAE=mean_absolute_error(y,p),
                         RMSE=np.sqrt(mean_squared_error(y,p)),bias=np.mean(p-y))
res={}
for name,mk in [("RandomForest",lambda:RandomForestRegressor(500,min_samples_leaf=2,random_state=42,n_jobs=-1)),
                ("Ridge",lambda:make_pipeline(StandardScaler(),Ridge(alpha=10)))]:
    m=mk(); m.fit(Xtr,ytr); p=m.predict(te[FEATS].values); res[name]=(p,sc(te.ET_closed_mm.values,p))
# baseline
bmean=sc(te.ET_closed_mm.values,np.full(len(te),ytr.mean()))
print(f"{'model':<16}{'R2':>8}{'MAE':>7}{'RMSE':>7}{'bias':>7}")
print("-"*45)
print(f"{'train-mean base':<16}{bmean['R2']:>8.3f}{bmean['MAE']:>7.2f}{bmean['RMSE']:>7.2f}{bmean['bias']:>7.2f}")
for n,(p,s) in res.items():
    print(f"{n:<16}{s['R2']:>8.3f}{s['MAE']:>7.2f}{s['RMSE']:>7.2f}{s['bias']:>7.2f}")
print("\nper test site (RF):")
prf=res["RandomForest"][0]; te2=te.assign(pred=prf)
for s in TEST:
    d=te2[te2.SITE_ID==s]
    print(f"  {s}: R2={r2_score(d.ET_closed_mm,d.pred):>6.3f}  MAE={mean_absolute_error(d.ET_closed_mm,d.pred):.2f}  n={len(d)}")
# figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
COL={"US-Elm":"#4a3aa7","US-EvM":"#1baf7a"}
fig,axes=plt.subplots(1,2,figsize=(9,4.2)); fig.patch.set_facecolor("#fcfcfb")
for ax,s in zip(axes,TEST):
    d=te2[te2.SITE_ID==s]; c=COL[s]; ax.set_facecolor("#fcfcfb")
    ax.scatter(d.ET_closed_mm,d.pred,s=8,color=c,alpha=0.3,edgecolors="none")
    lim=[0,8]; ax.plot(lim,lim,ls="--",color="#52514e",lw=1,alpha=.6); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.text(0.05,0.93,s,transform=ax.transAxes,fontsize=12,fontweight="bold",color=c,va="top")
    ax.text(0.05,0.84,f"R²={r2_score(d.ET_closed_mm,d.pred):.2f}\nMAE={mean_absolute_error(d.ET_closed_mm,d.pred):.2f}\nn={len(d)}",
            transform=ax.transAxes,fontsize=9,color="#52514e",va="top")
    ax.set_xlabel("observed ET (mm/d)",fontsize=9,color="#52514e")
    if ax is axes[0]: ax.set_ylabel("predicted ET (mm/d)",fontsize=9,color="#52514e")
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    ax.tick_params(length=0,labelsize=8,colors="#52514e")
fig.suptitle("Spatial upscaling — train Esm/TaS/Skr (2018–21), predict UNSEEN Elm & EvM",
             fontsize=12,fontweight="bold",color="#0b0b0b")
fig.subplots_adjust(top=0.86,bottom=0.13,wspace=0.2,left=0.09,right=0.98)
fig.savefig("/anvil/scratch/x-jwang120/coastal-et/figures/spatial_transfer.png",dpi=160,facecolor="#fcfcfb")
print("\nwrote spatial_transfer.png")
