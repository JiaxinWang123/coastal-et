import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
t=pd.read_csv("/anvil/scratch/x-jwang120/coastal-et/data/processed/physics_models.csv")
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
 "font.size":8,"axes.linewidth":0.7,"xtick.major.width":0.7,"ytick.major.width":0.7,
 "xtick.major.size":3,"ytick.major.size":3})
INK,GREY="#1a1a1a","#8a8a8a"
schemes=[("kfold","K-fold"),("leave_year","Leave-year-out"),("leave_site","Leave-site-out")]
# color: plain=dark, energy-balance/PT trees=teal, PINN=orange/red
CMAP={"Plain-ET (tree)":"#2C5F8A","EF-constrained (tree)":"#55A868",
 "PT-constrained (tree)":"#3B8C6E","PINN-PT (alpha)":"#DD8452","PINN-EB (LE,H soft)":"#C44E52"}
fig,ax=plt.subplots(figsize=(6.6,3.6),constrained_layout=True); fig.patch.set_facecolor("white")
x=np.arange(3); n=len(t); w=0.15
for i,(_,r) in enumerate(t.iterrows()):
    vals=[np.clip(r[c],-0.7,1) for c,_ in schemes]
    ax.bar(x+(i-n/2+0.5)*w, vals, w, color=CMAP.get(r.model,"#999"), label=r.model, zorder=3)
ax.axhline(0,color=GREY,lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([s[1] for s in schemes],fontsize=8.5)
ax.set_ylabel("$R^2$",fontsize=10); ax.set_ylim(-0.7,0.95); ax.set_yticks([-0.5,0,0.5])
for sp in ("top","right"): ax.spines[sp].set_visible(False)
ax.tick_params(colors=INK,labelsize=8)
ax.legend(frameon=False,fontsize=6.8,loc="lower left",ncol=1)
ax.set_title("Physics-constrained vs plain ML (13 coastal wetlands)",fontsize=10,fontweight="bold",color=INK)
for ext in ("png","pdf"):
    fig.savefig(f"/anvil/scratch/x-jwang120/coastal-et/figures/physics_models.{ext}",dpi=450,facecolor="white",bbox_inches="tight")
print("wrote physics_models.png (+ .pdf)")
