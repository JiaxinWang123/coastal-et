"""Validation figure for the gap-filling model: predicted-vs-observed + series."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R="/anvil/scratch/x-jwang120/coastal-et"; OUT=f"{R}/figures"; P=f"{R}/data/processed"
SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; MUT="#9a9a95"; FILL="#e0673a"
SITES=[("US-Elm","Fresh long-hydro","#4a3aa7"),("US-Esm","Fresh short-hydro","#008300"),
       ("US-TaS","Taylor Slough","#eda100"),("US-EvM","Saltwater marsh","#1baf7a"),
       ("US-Skr","Mangrove","#2a78d6")]
val=pd.read_csv(f"{P}/gapfill_validation.csv").set_index("site")
pred=pd.read_parquet(f"{P}/gapfill_loyo_predictions.parquet")

# ---- Fig A: predicted vs observed, per site ----
fig,axes=plt.subplots(1,5,figsize=(17,3.8)); fig.patch.set_facecolor(SURF)
for ax,(sid,lab,c) in zip(axes,SITES):
    ax.set_facecolor(SURF)
    d=pred[pred.SITE_ID==sid]
    if len(d):
        ax.scatter(d.ET_closed_mm,d.pred,s=6,color=c,alpha=0.25,edgecolors="none")
        lim=[0,max(d.ET_closed_mm.max(),d.pred.max())*1.05]
        ax.plot(lim,lim,color=INK2,lw=1,ls="--",alpha=0.6)
        ax.set_xlim(lim); ax.set_ylim(lim)
        r=val.loc[sid]
        ax.text(0.05,0.93,f"{sid}",transform=ax.transAxes,fontsize=11,fontweight="bold",color=c,va="top")
        ax.text(0.05,0.83,f"R²={r.R2:.2f}\nMAE={r.MAE:.2f}\nn={int(r.n)}",
                transform=ax.transAxes,fontsize=8.5,color=INK2,va="top")
    ax.set_xlabel("observed ET (mm/d)",fontsize=8,color=INK2)
    if ax is axes[0]: ax.set_ylabel("predicted ET (mm/d)",fontsize=8,color=INK2)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    for sp in ("left","bottom"): ax.spines[sp].set_color("#d8d8d4")
    ax.tick_params(length=0,labelsize=7,colors=INK2)
fig.suptitle("Gap-filling validation — leave-one-year-out, predicted vs observed daily ET",
             x=0.5,y=1.0,fontsize=13,fontweight="bold",color=INK)
fig.text(0.5,0.90,"Dashed = 1:1 line. Each point is one held-out day.",ha="center",fontsize=9,color=INK2)
fig.subplots_adjust(top=0.82,bottom=0.14,left=0.045,right=0.99,wspace=0.22)
fig.savefig(f"{OUT}/gapfill_validation.png",dpi=160,facecolor=SURF); print("wrote gapfill_validation.png")

# ---- Fig B: delivered continuous series, measured vs filled ----
fig,axes=plt.subplots(5,1,figsize=(14,10),sharex=True); fig.patch.set_facecolor(SURF)
for ax,(sid,lab,c) in zip(axes,SITES):
    ax.set_facecolor(SURF)
    o=pd.read_parquet(f"{P}/gapfilled_{sid}.parquet"); o.index=pd.to_datetime(o.index,utc=True)
    meas=o[o.ET_measured.notna()]; fil=o[o.is_filled]
    ax.plot(meas.index,meas.ET_measured,ls="",marker=".",ms=1.6,color=c,alpha=0.5,label="measured")
    ax.plot(fil.index,fil.ET_predicted,ls="",marker=".",ms=2.2,color=FILL,alpha=0.75,label="gap-filled")
    nm=int(o.ET_measured.notna().sum()); nf=int(o.is_filled.sum())
    ax.text(0.006,0.9,f"{sid}",transform=ax.transAxes,fontsize=11,fontweight="bold",color=c,va="top")
    ax.text(0.998,0.9,f"measured {nm}  ·  filled {nf}  ·  {100*(nm+nf)/2191:.0f}% coverage",
            transform=ax.transAxes,fontsize=8.5,color=MUT,va="top",ha="right")
    ax.set_ylim(0,9); ax.set_yticks([0,3,6,9])
    ax.set_xlim(pd.Timestamp("2018-01-01",tz="UTC"),pd.Timestamp("2024-01-01",tz="UTC"))
    ax.grid(axis="y",color="#e9e9e5",lw=0.7); ax.set_axisbelow(True)
    for sp in ("top","right","left"): ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#d8d8d4"); ax.tick_params(length=0,labelsize=8,colors=INK2)
axes[0].legend(loc="upper right",frameon=False,fontsize=9,bbox_to_anchor=(0.86,1.18),ncol=2)
axes[2].set_ylabel("daily ET  (mm/day)",fontsize=11,color=INK2)
fig.suptitle("Gap-filled daily ET, 2018–2023 — measured + reconstructed",
             x=0.008,y=0.995,ha="left",fontsize=14,fontweight="bold",color=INK)
fig.text(0.008,0.965,"Orange = model-reconstructed on days the tower did not measure "
         "(LOYO R²≈0.59). Only days with complete met+NDVI predictors are filled.",
         ha="left",fontsize=9.5,color=INK2)
fig.subplots_adjust(top=0.93,bottom=0.05,left=0.05,right=0.99,hspace=0.25)
fig.savefig(f"{OUT}/gapfill_series.png",dpi=160,facecolor=SURF); print("wrote gapfill_series.png")
