"""Site x month ET heatmap, using the domain-standard ET ramp.

WHY THE RAMP LIVES HERE AND NOT ON THE LINE PLOT

The palette the team supplied (tan -> yellow -> green -> teal -> blue) is the
conventional ET colour ramp. It encodes MAGNITUDE. That makes it right for this
figure -- every cell is inked, colour IS the ET value, and a colorbar states the
mapping -- and wrong for the five-site line plot, where the y-axis already is ET
and colouring lines by an ET ramp would invite readers to read identity as value.

Known flaws of the ramp, stated rather than hidden:
  * lightness is NOT monotone (rises to L 0.96 at the pale yellow, falls to 0.33
    at the dark blue), so low-ET tan and mid-ET green read as similarly light in
    greyscale or under some CVD;
  * the pale yellow sits at 1.09:1 against a light surface -- invisible if any
    background shows through.
Both are neutralised here: the grid is fully inked (no surface gaps), a colorbar
carries the mapping, and a CSV table view ships alongside.

Run via Slurm, never on the login node.
"""

import os

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

R = "/anvil/scratch/x-jwang120/coastal-et"
OUT = f"{R}/figures"

# the team's ET ramp, low -> high
ET_PALETTE = ["#DEC29B", "#E6CDA1", "#EDD9A6", "#F5E4A9", "#FFF4AD",
              "#C3E683", "#6BCC5C", "#3BB369", "#20998F", "#1C8691",
              "#16678A", "#114982", "#0B2C7A"]

SITES = [
    ("US-Elm", "Freshwater, long hydroperiod"),
    ("US-Esm", "Freshwater, short hydroperiod"),
    ("US-TaS", "Taylor Slough"),
    ("US-EvM", "Saltwater-intrusion marsh"),
    ("US-Skr", "Mangrove (Shark River)"),
]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8a85"
NODATA = "#ececea"        # a gap must look like a gap, not like low ET


def main():
    os.makedirs(OUT, exist_ok=True)
    cmap = LinearSegmentedColormap.from_list("et", ET_PALETTE, N=256)
    cmap.set_bad(NODATA)

    et = pd.read_parquet(f"{R}/data/processed/daily_closed_et.parquet")
    et.index = pd.to_datetime(et.index, utc=True)

    months = pd.period_range("2008-01", "2025-12", freq="M")
    grid = pd.DataFrame(index=[s for s, _ in SITES], columns=months, dtype=float)

    for sid, _ in SITES:
        d = et[(et.SITE_ID == sid) & (et.ET_closed_mm.notna())]
        if d.empty:
            continue
        m = d.ET_closed_mm.groupby(d.index.to_period("M")).agg(["mean", "size"])
        # a month with only a handful of days is not a monthly mean
        m = m[m["size"] >= 10]["mean"]
        for p, v in m.items():
            if p in grid.columns:
                grid.loc[sid, p] = v

    arr = np.ma.masked_invalid(grid.values.astype(float))

    fig, ax = plt.subplots(figsize=(15, 3.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    vmin, vmax = 1.0, 6.0
    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")

    ax.set_yticks(range(len(SITES)))
    ax.set_yticklabels([f"{s}   {d}" for s, d in SITES], fontsize=10, color=INK)
    yr_start = [i for i, p in enumerate(months) if p.month == 1]
    ax.set_xticks(yr_start)
    ax.set_xticklabels([months[i].year for i in yr_start], fontsize=9.5, color=INK2)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # mark the 2022-23 analysis window without occluding the data
    for edge in ("2022-01", "2024-01"):
        x = months.get_loc(pd.Period(edge, freq="M"))
        ax.axvline(x - 0.5, color=INK, lw=1.1, alpha=0.55)

    cb = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.03, extend="both")
    cb.set_label("Closed ET  (mm day$^{-1}$)", fontsize=10, color=INK2)
    cb.ax.tick_params(labelsize=9, colors=INK2, length=0)
    cb.outline.set_visible(False)

    ax.set_title("Monthly mean closed ET across the Everglades gradient",
                 loc="left", fontsize=14, fontweight="bold", color=INK, pad=14)
    fig.text(0.005, 0.885,
             "Grey = no data (≥ 10 valid days required for a monthly mean). "
             "Vertical rules bracket the 2022–23 analysis window.",
             fontsize=9.5, color=MUTED)

    fig.subplots_adjust(top=0.79, bottom=0.10, left=0.20, right=0.99)
    p = f"{OUT}/et_heatmap_everglades.png"
    fig.savefig(p, dpi=170, facecolor=SURFACE)
    print(f"wrote {p}")

    grid.round(2).to_csv(f"{OUT}/et_monthly_table.csv")
    print(f"wrote {OUT}/et_monthly_table.csv  (the table view)")

    cov = (~np.ma.getmaskarray(arr)).sum(axis=1)
    print("\nmonths with a valid mean, per site:")
    for (s, _), n in zip(SITES, cov):
        print(f"  {s}: {n:>3} / {len(months)}")


if __name__ == "__main__":
    main()
