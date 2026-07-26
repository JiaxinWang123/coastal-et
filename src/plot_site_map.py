"""Clean distribution map of the 13 coastal-wetland flux towers.

CONUS basemap (Census states); sites coloured by region and sized by record length;
a Florida inset resolves the 5-site Everglades cluster. Replaces the earlier crowded map.
"""
import os, glob
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, ConnectionPatch
from matplotlib.lines import Line2D

R = "/anvil/scratch/x-jwang120/coastal-et"
FIG = f"{R}/figures"

# lon, lat, region
SITES = {
    "US-EDN": (-122.1140, 37.6156, "Pacific delta"),
    "US-Myb": (-121.7650, 38.0499, "Pacific delta"),
    "US-Tw4": (-121.6413, 38.1027, "Pacific delta"),
    "US-LA3": (-89.9153, 29.4936, "Gulf"),
    "US-HB1": (-79.1957, 33.3455, "Atlantic"),
    "US-HB4": (-79.2975, 33.2015, "Atlantic"),
    "US-NC4": (-75.9135, 35.7817, "Atlantic"),
    "US-StJ": (-75.4372, 39.0882, "Atlantic"),
    "US-TaS": (-80.6391, 25.1908, "Everglades"),
    "US-EvM": (-80.3810, 25.3539, "Everglades"),
    "US-Skr": (-81.0776, 25.4108, "Everglades"),
    "US-Esm": (-80.5946, 25.4379, "Everglades"),
    "US-Elm": (-80.7826, 25.5519, "Everglades"),
}
COL = {"Pacific delta": "#4C72B0", "Gulf": "#DD8452", "Atlantic": "#55A868", "Everglades": "#C44E52"}


def states():
    cache = f"{R}/data/raw/cb_2022_us_state_20m.shp"
    g = gpd.read_file(cache)
    return g[~g.STUSPS.isin({"AK", "HI", "PR", "GU", "MP", "AS", "VI"})].to_crs(4326)


def main():
    st = states()
    tab = pd.read_csv(f"{FIG}/et_timeseries_13sites_table.csv").set_index("site")
    days = {s: int(tab.days.get(s, 1500)) for s in SITES}
    def msize(d):
        return 60 + 340 * (d - 1000) / 5000        # record length -> marker area

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 9})
    fig = plt.figure(figsize=(12, 7.6)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.85])
    st.plot(ax=ax, color="#f4f4f2", edgecolor="#cfcfcf", lw=0.5, zorder=0)
    ax.set_xlim(-125, -66); ax.set_ylim(23, 50)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.text(0.03, 0.965, "Coastal-wetland flux-tower network (13 sites)",
             fontsize=15, fontweight="bold", color="#111")
    fig.text(0.03, 0.925, "AmeriFlux/FLUXNET towers spanning Pacific delta, Gulf, Atlantic "
             "and the Everglades. Marker size = record length.", fontsize=10, color="#555")

    ev = {s: v for s, v in SITES.items() if v[2] == "Everglades"}
    non = {s: v for s, v in SITES.items() if v[2] != "Everglades"}

    # --- helper: a zoom box (inset) for a crowded cluster ---
    def cluster_inset(sub, rect, xlim, ylim, title, color, anchor, laboff=None):
        laboff = laboff or {}
        clo = np.mean([v[0] for v in sub.values()]); cla = np.mean([v[1] for v in sub.values()])
        ax.scatter(clo, cla, s=95, color=color, edgecolor="white", lw=1.2, zorder=6)
        axin = fig.add_axes(rect)
        st.plot(ax=axin, color="#f0f0ee", edgecolor="#cfcfcf", lw=0.5)
        axin.set_xlim(*xlim); axin.set_ylim(*ylim)
        for s, (lo, la, reg) in sub.items():
            axin.scatter(lo, la, s=msize(days[s]), color=color, edgecolor="white", lw=1.1, zorder=5)
            dx, dy, ha = laboff.get(s, (4, 4, "left"))
            axin.annotate(s, (lo, la), xytext=(dx, dy), textcoords="offset points",
                          fontsize=8, fontweight="bold", color="#222", ha=ha, va="center", zorder=7)
        axin.set_xticks([]); axin.set_yticks([])
        for sp in axin.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(1.4)
        axin.set_title(title, fontsize=8.5, fontweight="bold", color=color)
        fig.add_artist(ConnectionPatch((clo, cla), anchor, "data", "axes fraction",
                                       axesA=ax, axesB=axin, color=color, lw=0.9, alpha=0.7))

    # well-separated Gulf + Atlantic sites: leader-line labels on the main map
    main_sites = {s: v for s, v in SITES.items() if v[2] in ("Gulf", "Atlantic")}
    LAB = {"US-HB1": (16, 11), "US-HB4": (16, -13), "US-NC4": (14, 0),
           "US-StJ": (14, 2), "US-LA3": (12, -12)}
    for s, (lo, la, reg) in main_sites.items():
        ax.scatter(lo, la, s=msize(days[s]), color=COL[reg], edgecolor="white", lw=1.1, zorder=5)
        dx, dy = LAB[s]
        ax.annotate(s, (lo, la), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8.5, fontweight="bold", ha="left" if dx > 0 else "right",
                    va="center", color="#222", zorder=7,
                    arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.6, shrinkA=0, shrinkB=3))

    # two zoom boxes for the crowded clusters
    ca = {s: v for s, v in SITES.items() if v[2] == "Pacific delta"}
    cluster_inset(ca, [0.055, 0.585, 0.205, 0.26], (-122.5, -121.3), (37.25, 38.45),
                  "Sacramento–San Joaquin delta (3 sites)", COL["Pacific delta"], (0.5, 0.0),
                  laboff={"US-Tw4": (6, 9, "left"), "US-Myb": (-6, -10, "right"), "US-EDN": (7, 0, "left")})
    cluster_inset(ev, [0.135, 0.075, 0.25, 0.32], (-81.35, -80.15), (24.95, 25.75),
                  "Everglades cluster (5 sites)", COL["Everglades"], (0.5, 1.0),
                  laboff={"US-Esm": (11, 2, "left"), "US-Skr": (7, 0, "left"), "US-Elm": (7, 6, "left"),
                          "US-EvM": (7, -2, "left"), "US-TaS": (7, -6, "left")})

    # legend: region colours + size key
    reg_handles = [Line2D([0], [0], marker="o", ls="", mfc=COL[r], mec="white", ms=10,
                          label=f"{r} ({sum(1 for v in SITES.values() if v[2]==r)})")
                   for r in ["Pacific delta", "Gulf", "Atlantic", "Everglades"]]
    leg1 = ax.legend(handles=reg_handles, title="Region", loc="upper right",
                     frameon=False, fontsize=9, title_fontsize=9.5)
    ax.add_artist(leg1)
    size_handles = [Line2D([0], [0], marker="o", ls="", mfc="#999", mec="white",
                           ms=np.sqrt(msize(d)), label=f"{d:,} d") for d in [1500, 3500, 5500]]
    ax.legend(handles=size_handles, title="Record length", loc="lower right",
              frameon=False, fontsize=8.5, title_fontsize=9, labelspacing=1.4, borderpad=1.0)

    fig.savefig(f"{FIG}/site_map.png", dpi=200, facecolor="white")
    print("wrote", f"{FIG}/site_map.png")


if __name__ == "__main__":
    main()
