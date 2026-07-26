"""Key result figure: 5 sites vs 13 sites, leave-tower-out ET prediction."""
import sys; sys.path.insert(0,"/anvil/scratch/x-jwang120/coastal-et/src")
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import more_sites as M
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

d=pd.read_parquet("/anvil/scratch/x-jwang120/coastal-et/data/processed/more_sites_table.parquet")
FE=M.FEATS
def loto(data,sites):
    P=[]
    for s in sites:
        tr,te=data[data.SITE_ID!=s],data[data.SITE_ID==s]
        if len(te)<5: continue
        m=ExtraTreesRegressor(400,min_samples_leaf=3,random_state=0,n_jobs=-1).fit(tr[FE].values,tr.ET_closed_mm.values)
        P.append(te.assign(pred=m.predict(te[FE].values)))
    return pd.concat(P)
ev=[s for s in M.EVERGLADES if s in d.SITE_ID.unique()]
g5=loto(d[d.SITE_ID.isin(ev)],ev)
g13=loto(d,sorted(d.SITE_ID.unique()))

plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
  "font.size":8,"axes.linewidth":0.7,"xtick.major.width":0.7,"ytick.major.width":0.7,
  "xtick.major.size":3,"ytick.major.size":3})
INK,GREY="#1a1a1a","#8a8a8a"
# colour by ecosystem/region group (cleaner + meaningful)
GROUP={"US-Esm":"Everglades (FL)","US-TaS":"Everglades (FL)","US-Skr":"Everglades (FL)",
 "US-Elm":"Everglades (FL)","US-EvM":"Everglades (FL)",
 "US-HB1":"Atlantic salt marsh (SC)","US-HB4":"Atlantic salt marsh (SC)","US-StJ":"Atlantic salt marsh (DE)",
 "US-LA3":"Gulf salt marsh (LA)","US-NC4":"Tidal forest (NC)",
 "US-EDN":"Pacific delta (CA)","US-Myb":"Pacific delta (CA)","US-Tw4":"Pacific delta (CA)"}
GCOL={"Everglades (FL)":"#4C72B0","Atlantic salt marsh (SC)":"#DD8452",
 "Atlantic salt marsh (DE)":"#E0A458","Gulf salt marsh (LA)":"#C44E52",
 "Tidal forest (NC)":"#55A868","Pacific delta (CA)":"#8172B3"}
COL={s:GCOL[GROUP[s]] for s in d.SITE_ID.unique()}
fig,axes=plt.subplots(1,2,figsize=(7.4,3.9),constrained_layout=True); fig.patch.set_facecolor("white")
for ax,(g,tag,title,sub) in zip(axes,[(g5,"a","5 Everglades sites","spectrally similar"),
                                       (g13,"b","13 coastal wetlands","diverse ecosystems")]):
    ax.set_facecolor("white"); ax.set_aspect("equal")
    ax.plot([0,8],[0,8],ls=(0,(4,3)),color=GREY,lw=0.8,zorder=1)
    for s in g.SITE_ID.unique():
        gg=g[g.SITE_ID==s]; ax.scatter(gg.ET_closed_mm,gg.pred,s=9,color=COL[s],alpha=0.7,
            edgecolors="white",linewidths=0.2,zorder=3)
    x,y=g.ET_closed_mm.values,g.pred.values; b1,b0=np.polyfit(x,y,1)
    ax.plot([0,8],[b0,b0+8*b1],color=INK,lw=1.1,zorder=2)
    r2=r2_score(y,x*0+y) if False else r2_score(x,y)
    r2=r2_score(g.ET_closed_mm,g.pred); rmse=np.sqrt(mean_squared_error(g.ET_closed_mm,g.pred))
    ax.text(0.05,0.96,f"$R^2$ = {r2:.2f}\nRMSE = {rmse:.2f}\nslope = {b1:.2f}\n"
            f"{g.SITE_ID.nunique()} sites, $n$={len(g)}",transform=ax.transAxes,fontsize=7.2,
            va="top",color=INK,linespacing=1.35)
    if tag=="a": ax.text(0.97,0.06,sub,transform=ax.transAxes,fontsize=6.8,ha="right",va="bottom",color=GREY,style="italic")
    ax.set_title(title,fontsize=8.5,color=INK,pad=3)
    ax.text(-0.16,1.06,tag,transform=ax.transAxes,fontsize=11,fontweight="bold",va="top",color=INK)
    ax.set_xlim(0,8); ax.set_ylim(0,8); ax.set_xticks([0,2,4,6,8]); ax.set_yticks([0,2,4,6,8])
    ax.set_xlabel("Observed ET (mm d$^{-1}$)",fontsize=8)
    if ax is axes[0]: ax.set_ylabel("Predicted ET (mm d$^{-1}$)",fontsize=8)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK,labelsize=7)
from matplotlib.lines import Line2D
seen=[g for g in ["Everglades (FL)","Atlantic salt marsh (SC)","Atlantic salt marsh (DE)",
 "Gulf salt marsh (LA)","Tidal forest (NC)","Pacific delta (CA)"]]
leg=[Line2D([0],[0],marker="o",ls="",mfc=GCOL[g],mec="white",ms=5,label=g) for g in seen]
axes[1].legend(handles=leg,frameon=False,fontsize=5.6,loc="lower right",
    handletextpad=0.2,labelspacing=0.3,borderpad=0.2)
fig.suptitle("Spatial upscaling to an UNSEEN tower — site diversity is the key",
             fontsize=10.5,fontweight="bold",color=INK)
for ext in ("png","pdf"):
    fig.savefig(f"/anvil/scratch/x-jwang120/coastal-et/figures/upscaling_more_sites.{ext}",
                dpi=450,facecolor="white",bbox_inches="tight")
print(f"5-site LOTO R2={r2_score(g5.ET_closed_mm,g5.pred):.3f}  13-site LOTO R2={r2_score(g13.ET_closed_mm,g13.pred):.3f}")
print("wrote upscaling_more_sites.png (+ .pdf)")
