"""Time series of LAI, LST and vegetation/water indices, 2018-2023, five sites.

Footprint-weighted per-overpass values from the per-pixel Landsat record.
Small multiples: one panel per index, five site series (points; sparse by nature).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/anvil/scratch/x-jwang120/coastal-et"
PIX = f"{R}/data/interim/pixels"
OUT = f"{R}/figures"
SITES = [("US-Elm", "#4a3aa7"), ("US-Esm", "#008300"), ("US-TaS", "#eda100"),
         ("US-EvM", "#1baf7a"), ("US-Skr", "#2a78d6")]
SURF = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"
# (column, label, unit, y-limits)
VARS = [("LAI", "LAI", "m²/m²", (0, 6)),
        ("LST_C", "Land surface temp", "°C", (10, 45)),
        ("NDVI", "NDVI", "-", (-0.1, 0.9)),
        ("EVI2", "EVI2", "-", (-0.1, 0.9)),
        ("NDWI", "NDWI (NIR-SWIR)", "-", (-0.6, 0.6)),
        ("MNDWI", "MNDWI (Green-SWIR)", "-", (-0.6, 0.6))]


def wavg(x, w):
    w = np.where(np.isfinite(x), w, 0.0)
    x = np.where(np.isfinite(x), x, 0.0)
    return x.dot(w) / w.sum() if w.sum() > 0 else np.nan


def site_series(site):
    px = pd.read_parquet(f"{PIX}/{site}_pixels.parquet")
    px["date"] = pd.to_datetime(px["date"], utc=True)
    savi = px["SAVI"].clip(0, 0.685)
    px["LAI"] = (-np.log((0.69 - savi) / 0.59) / 0.91).clip(0, 6)
    px["LST_C"] = px["LST_K"] - 273.15
    cols = ["LAI", "LST_C", "NDVI", "EVI2", "NDWI", "MNDWI"]
    g = px.groupby("date").apply(
        lambda x: pd.Series({c: wavg(x[c].values, x.fp_weight.values) for c in cols}))
    return g.sort_index()


def main():
    data = {s: site_series(s) for s, _ in SITES}
    n = len(VARS)
    fig, axes = plt.subplots(n, 1, figsize=(14, 13), sharex=True)
    fig.patch.set_facecolor(SURF)
    x0 = pd.Timestamp("2018-01-01", tz="UTC")
    x1 = pd.Timestamp("2024-01-01", tz="UTC")
    for ax, (col, lab, unit, ylim) in zip(axes, VARS):
        ax.set_facecolor(SURF)
        for s, c in SITES:
            d = data[s][col].dropna()
            if len(d):
                ax.plot(d.index, d.values, ls="", marker="o", ms=3.2, color=c,
                        alpha=0.55, label=s)
                # light 60-day rolling median to guide the eye
                roll = d.rolling("90D").median()
                ax.plot(roll.index, roll.values, lw=1.2, color=c, alpha=0.85)
        ax.set_ylim(ylim)
        ax.set_xlim(x0, x1)
        ax.set_ylabel(f"{lab}\n({unit})", fontsize=9.5, color=INK, rotation=90,
                      labelpad=8)
        ax.grid(axis="y", color="#ececea", lw=0.7)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#d8d8d4")
        ax.tick_params(length=0, labelsize=8.5, colors=INK2)
    axes[0].legend(frameon=False, fontsize=9, ncol=5, loc="upper center",
                   bbox_to_anchor=(0.5, 1.42))
    fig.suptitle("Biophysical predictors from Landsat — footprint-weighted, "
                 "2018–2023", x=0.5, y=0.995, fontsize=15, fontweight="bold", color=INK)
    fig.text(0.5, 0.965, "Points = per-overpass; line = 90-day rolling median. "
             "LAI from SAVI (Anderson et al.). Sparse where clouds/revisit limit coverage.",
             ha="center", fontsize=9.5, color=INK2)
    fig.subplots_adjust(top=0.93, bottom=0.03, left=0.07, right=0.99, hspace=0.18)
    fig.savefig(f"{OUT}/indices_timeseries.png", dpi=150, facecolor=SURF)
    print(f"wrote {OUT}/indices_timeseries.png")
    # quick per-site medians table
    print("\nmedian over record:")
    print(f"{'site':<8}" + "".join(f"{v[0]:>9}" for v in VARS))
    for s, _ in SITES:
        print(f"{s:<8}" + "".join(f"{data[s][v[0]].median():>9.2f}" for v in VARS))


if __name__ == "__main__":
    main()
