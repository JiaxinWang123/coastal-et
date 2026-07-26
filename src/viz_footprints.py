"""Combined 5-site Kljun-2015 footprint climatology figure."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
F="/anvil/scratch/x-jwang120/coastal-et/data/interim/footprint"
OUT="/anvil/scratch/x-jwang120/coastal-et/figures"
SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; MUT="#9a9a95"
SITES=[("US-Elm","Fresh long-hydro  (zm 4 m)","#4a3aa7"),
       ("US-Esm","Fresh short-hydro  (zm 4 m)","#008300"),
       ("US-TaS","Taylor Slough  (zm 4 m)","#eda100"),
       ("US-EvM","Saltwater marsh  (zm 4 m)","#1baf7a"),
       ("US-Skr","Mangrove  (zm 27 m)","#2a78d6")]
EXT=250.0   # common half-extent (m) so footprint SIZES are comparable across sites

def contour_levels(f):
    """Return density thresholds enclosing 50% and 80% of the source area."""
    flat=np.sort(f[np.isfinite(f)])[::-1]; cum=np.cumsum(flat)
    lv={}
    for pct in (0.5,0.8):
        i=np.searchsorted(cum,pct)
        lv[pct]=flat[min(i,len(flat)-1)]
    return lv

fig,axes=plt.subplots(1,5,figsize=(17.5,4.0)); fig.patch.set_facecolor(SURF)
for ax,(sid,lab,c) in zip(axes,SITES):
    d=np.load(f"{F}/{sid}_ffp.npz")
    f=d["fclim"]; x=d["x"]; y=d["y"]
    # x,y may be 1-D or 2-D; build extent
    xv=x[0] if x.ndim==2 else x; yv=y[:,0] if y.ndim==2 else y
    ext=[xv.min(),xv.max(),yv.min(),yv.max()]
    cmap=LinearSegmentedColormap.from_list(sid,["#ffffff",c],N=256)
    ax.set_facecolor("#ffffff")
    fp=np.where(f>0,f,np.nan)
    ax.imshow(fp,origin="lower",extent=ext,cmap=cmap,aspect="equal",
              vmax=np.nanpercentile(fp,99.5))
    lv=contour_levels(f)
    ax.contour(f,levels=[lv[0.8],lv[0.5]],extent=ext,colors=[INK2,INK],
               linewidths=[0.8,1.1],alpha=0.8)
    ax.plot(0,0,marker="^",ms=9,color=c,mec="white",mew=1.2,zorder=5)  # tower
    # 80% source area
    flat=np.sort(f[np.isfinite(f)])[::-1]; cum=np.cumsum(flat)
    area80=int((f>=flat[np.searchsorted(cum,0.8)]).sum())*100/1e4
    ax.set_xlim(-EXT,EXT); ax.set_ylim(-EXT,EXT)
    ax.set_title(sid,fontsize=12,fontweight="bold",color=c,pad=3)
    ax.text(0.5,-0.16,lab,transform=ax.transAxes,ha="center",fontsize=8.5,color=INK2)
    ax.text(0.04,0.96,f"80% area\n{area80:.1f} ha",transform=ax.transAxes,fontsize=8,
            color=INK2,va="top")
    ax.set_xticks([-200,0,200]); ax.set_yticks([-200,0,200])
    ax.tick_params(length=0,labelsize=7,colors=INK2)
    for sp in ax.spines.values(): sp.set_color("#d8d8d4")
    if ax is axes[0]: ax.set_ylabel("N–S distance (m)",fontsize=8.5,color=INK2)

fig.suptitle("Flux-footprint climatology (Kljun et al. 2015) — 2018–2023, "
             "same 500 m extent",x=0.5,y=1.0,fontsize=14,fontweight="bold",color=INK)
fig.text(0.5,0.90,"Shading = footprint weight; lines = 50% (dark) & 80% (grey) source "
         "areas; ▲ = tower. Mangrove (tall tower) integrates a far larger area than "
         "the 4 m marsh towers.",ha="center",fontsize=9,color=INK2)
fig.subplots_adjust(top=0.80,bottom=0.20,left=0.045,right=0.99,wspace=0.16)
fig.savefig(f"{OUT}/footprint_climatology.png",dpi=170,facecolor=SURF)
print("wrote footprint_climatology.png")
