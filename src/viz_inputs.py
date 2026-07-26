"""Visualize ALL model inputs across the 5 sites, 2018-2023.

Fig 1  input_data_overview.png    variable (rows) x site (cols) time-series grid
Fig 2  input_data_availability.png per-site observation timelines (shows sparsity)
Fig 3  input_distributions.png     per-feature distributions by site

Palette: validated 5-slot categorical set, same site->colour as the ET figure.
"""
import os, sys, glob; sys.path.insert(0,"/anvil/scratch/x-jwang120/coastal-et/src")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import train_with_ecostress as TE   # reuses its build() with ECOSTRESS + clean ERA5

R="/anvil/scratch/x-jwang120/coastal-et"; OUT=f"{R}/figures"; os.makedirs(OUT,exist_ok=True)
SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; MUT="#9a9a95"
# site -> (label, colour), gradient order, matching the ET time-series figure
SITES=[("US-Elm","Fresh long-hydro","#4a3aa7"),("US-Esm","Fresh short-hydro","#008300"),
       ("US-TaS","Taylor Slough","#eda100"),("US-EvM","Saltwater marsh","#1baf7a"),
       ("US-Skr","Mangrove","#2a78d6")]
COL={s:c for s,_,c in SITES}

data=TE.data.copy()            # daily feature table already built at import
data.index=pd.to_datetime(data.index,utc=True)
data=data[(data.index>="2018-01-01")&(data.index<="2023-12-31")]   # requested window
print(f"input table: {len(data)} daily rows, {data.SITE_ID.nunique()} sites, "
      f"{data.index.min().date()}..{data.index.max().date()}")

# variables to show: target + met + satellite, with nice labels & units
VARS=[("ET_closed_mm","ET (target)","mm/d",True),
      ("ETo_mm","Reference ET","mm/d",False),
      ("LST_K","Landsat LST","K",False),
      ("ECO_LST_K","ECOSTRESS LST","K",False),
      ("NDVI_fused","NDVI","-",False),
      ("TA_ERA","Air temp","C",False),
      ("VPD_ERA","VPD","kPa",False),
      ("SW_IN_ERA","Solar rad","W/m2",False)]
VARS=[v for v in VARS if v[0] in data.columns]

# ---------- Fig 1: variable x site grid ----------
nv,ns=len(VARS),len(SITES)
fig,axes=plt.subplots(nv,ns,figsize=(15,11),sharex=True)
fig.patch.set_facecolor(SURF)
x0,x1=pd.Timestamp("2018-01-01",tz="UTC"),pd.Timestamp("2024-01-01",tz="UTC")
for i,(col,lab,unit,tgt) in enumerate(VARS):
    vmax=np.nanpercentile(data[col],99); vmin=np.nanpercentile(data[col],1)
    for j,(sid,slab,c) in enumerate(SITES):
        ax=axes[i,j]; ax.set_facecolor(SURF)
        d=data[data.SITE_ID==sid][col].dropna()
        if len(d):
            # break the line over gaps > 20 d
            full=d.reindex(pd.date_range(d.index.min(),d.index.max(),freq="D",tz="UTC"))
            gap=full.isna().rolling(20,min_periods=1).sum()>=18
            sparse = col in ("LST_K","ECO_LST_K")   # thermal: dots, it's sparse
            if sparse:
                ax.plot(d.index,d.values,ls="",marker=".",ms=2,color=c,alpha=0.6)
            else:
                ax.plot(full.index,full.mask(gap).values,lw=0.9,color=c,alpha=0.9)
        ax.set_ylim(vmin-abs(vmin)*0.1-0.05, vmax+abs(vmax)*0.1+0.05)
        ax.set_xlim(x0,x1)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.tick_params(length=0,labelsize=6,colors=INK2)
        if i<nv-1: ax.set_xticklabels([])
        if j>0: ax.set_yticklabels([])
        if i==0: ax.set_title(sid,fontsize=9,color=c,fontweight="bold",pad=3)
        if j==0:
            ax.set_ylabel(f"{lab}\n{unit}",fontsize=8,color=INK,rotation=0,
                          ha="right",va="center",labelpad=26)
        if tgt: ax.set_facecolor("#f4f1e8")   # tint the target row
fig.suptitle("All model inputs — 5 Everglades sites, daily, 2018–2023",
             x=0.5,y=0.995,fontsize=14,fontweight="bold",color=INK)
fig.text(0.5,0.965,"Target row (ET) tinted. Thermal (LST) shown as points — it is "
         "sparse; lines break over data gaps >20 d.",ha="center",fontsize=9,color=INK2)
fig.subplots_adjust(top=0.94,bottom=0.04,left=0.10,right=0.99,hspace=0.25,wspace=0.08)
fig.savefig(f"{OUT}/input_data_overview.png",dpi=160,facecolor=SURF)
print("wrote input_data_overview.png")

# ---------- Fig 2: data availability ----------
STREAMS=[("ET_closed_mm","ET (target)"),("TA_ERA","ERA5 met"),("ETo_mm","Reference ET"),
         ("NDVI_fused","NDVI (LS+S2)"),("LST_K","Landsat LST"),("ECO_LST_K","ECOSTRESS LST")]
STREAMS=[s for s in STREAMS if s[0] in data.columns]
fig,ax=plt.subplots(figsize=(14,6)); fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
ylab=[]; ypos=[]; y=0
for sid,slab,c in SITES:
    d=data[data.SITE_ID==sid]
    for col,cl in STREAMS:
        dates=d[d[col].notna()].index
        if len(dates):
            ax.plot(dates,[y]*len(dates),ls="",marker="|",ms=7,color=c,
                    alpha=0.85 if col!="ET_closed_mm" else 0.5)
        ylab.append(f"{sid} · {cl}"); ypos.append(y); y+=1
    y+=0.8
ax.set_yticks(ypos); ax.set_yticklabels(ylab,fontsize=7,color=INK2)   # ACTUAL positions
ax.invert_yaxis()
ax.set_xlim(pd.Timestamp("2018-01-01",tz="UTC"),pd.Timestamp("2024-01-01",tz="UTC"))
for sp in ("top","right","left"): ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color("#d8d8d4"); ax.tick_params(length=0,colors=INK2)
ax.set_title("Input data availability — each tick is an observation day",
             loc="left",fontsize=13,fontweight="bold",color=INK,pad=10)
fig.text(0.008,0.93,"Met/ETo are daily-continuous; NDVI moderately dense; LST sparse. "
         "This sparsity is why thermal struggles as a daily predictor.",fontsize=9,color=INK2)
fig.subplots_adjust(top=0.88,bottom=0.06,left=0.16,right=0.99)
fig.savefig(f"{OUT}/input_data_availability.png",dpi=160,facecolor=SURF)
print("wrote input_data_availability.png")

# observation counts table
print("\n=== observation counts (2018-2023) ===")
print(f"{'site':<8}"+"".join(f"{c[:9]:>11}" for _,c in STREAMS))
for sid,_,_ in SITES:
    d=data[data.SITE_ID==sid]
    print(f"{sid:<8}"+"".join(f"{int(d[col].notna().sum()):>11}" for col,_ in STREAMS))
