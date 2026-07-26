"""Plot the closed daily ET time series for the 5 Everglades towers.

Form: small multiples, one panel per site, ordered along the ecological gradient
(freshwater -> brackish -> mangrove). Five overlapping daily series would be
spaghetti, and panels also make each site's very different record length -- which
is a real constraint on the modelling -- impossible to miss.

Palette: the validated 5-slot categorical set (CVD adjacent-pair dE 24.2, well
above the 12 floor). Two hues warn on contrast-vs-surface, so every panel carries
a direct text label: identity never rests on colour alone.

Run via Slurm, never on the login node.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = "/anvil/scratch/x-jwang120/coastal-et"
OUT = f"{R}/figures"

# ordered along the gradient the study is built to detect
SITES = [
    ("US-Elm", "Freshwater marsh, long hydroperiod",  "#4a3aa7"),
    ("US-Esm", "Freshwater marsh, short hydroperiod", "#008300"),
    ("US-TaS", "Taylor Slough / Panhandle",           "#eda100"),
    ("US-EvM", "Saltwater-intrusion marsh",           "#1baf7a"),
    ("US-Skr", "Mangrove forest (Shark River)",       "#2a78d6"),
]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8a85"


def main():
    import os
    os.makedirs(OUT, exist_ok=True)

    et = pd.read_parquet(f"{R}/data/processed/daily_closed_et.parquet")
    et.index = pd.to_datetime(et.index, utc=True)

    fig, axes = plt.subplots(
        len(SITES), 1, figsize=(13, 11), sharex=True,
        gridspec_kw=dict(hspace=0.32))
    fig.patch.set_facecolor(SURFACE)

    x0 = pd.Timestamp("2008-01-01", tz="UTC")
    x1 = pd.Timestamp("2026-01-01", tz="UTC")

    for ax, (sid, desc, colour) in zip(axes, SITES):
        ax.set_facecolor(SURFACE)
        d = et[(et.SITE_ID == sid) & (et.ET_closed_mm.notna())].sort_index()

        # the 2022-2023 analysis window
        ax.axvspan(pd.Timestamp("2022-01-01", tz="UTC"),
                   pd.Timestamp("2024-01-01", tz="UTC"),
                   color="#000000", alpha=0.04, lw=0, zorder=0)

        # daily values: faint, they are noisy by nature
        ax.plot(d.index, d.ET_closed_mm, lw=0, marker=".", ms=1.4,
                color=colour, alpha=0.28, zorder=2)

        # 30-day rolling mean carries the readable signal -- but it MUST be
        # broken at data gaps. Drawn naively it connects across multi-year
        # outages (US-Skr 2012-2018, US-Esm 2014-2016), painting a confident
        # flat line over ground truth that does not exist.
        daily = d.ET_closed_mm.reindex(
            pd.date_range(d.index.min(), d.index.max(), freq="D", tz="UTC"))
        roll = daily.rolling(30, min_periods=10).mean()
        gap = daily.isna().rolling(30, min_periods=1).sum() >= 25
        roll = roll.mask(gap)          # NaN breaks the line instead of bridging it
        ax.plot(roll.index, roll.values, lw=2.0, color=colour, zorder=3)

        # closure uncertainty, where FLUXNET publishes it
        if {"ET_lo_mm", "ET_hi_mm"} <= set(d.columns) and d.ET_lo_mm.notna().any():
            lo = d.ET_lo_mm.rolling(30, min_periods=10).mean()
            hi = d.ET_hi_mm.rolling(30, min_periods=10).mean()
            ax.fill_between(d.index, lo, hi, color=colour, alpha=0.14, lw=0, zorder=1)

        # direct label -- identity never rests on colour alone
        closure = d.CLOSURE.iloc[0] if len(d) else "-"
        nice = {"ONEFlux_LE_CORR": "ONEFlux closure",
                "FLUXNET_bowen_G0": "Bowen closure (G=0)",
                "BASE_bowen_ratio": "Bowen closure, BASE"}.get(closure, closure)
        ax.text(0.006, 0.93, sid, transform=ax.transAxes, fontsize=12,
                fontweight="bold", color=INK, va="top")
        ax.text(0.075, 0.925, desc, transform=ax.transAxes, fontsize=10,
                color=INK2, va="top")
        ax.text(0.998, 0.93,
                f"{len(d):,} days   mean {d.ET_closed_mm.mean():.2f} mm/d   {nice}",
                transform=ax.transAxes, fontsize=9, color=MUTED,
                va="top", ha="right")

        ax.set_ylim(0, 9)
        ax.set_yticks([0, 3, 6, 9])
        ax.set_xlim(x0, x1)
        ax.grid(axis="y", color="#e6e6e2", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color("#d8d8d4")
        ax.tick_params(colors=INK2, labelsize=9, length=0)

    axes[len(SITES) // 2].set_ylabel("Closed daily ET  (mm day$^{-1}$)",
                                     fontsize=11, color=INK2, labelpad=10)
    axes[-1].tick_params(labelsize=10)

    fig.suptitle("Energy-balance-closed daily ET across the Everglades gradient",
                 x=0.008, y=0.985, ha="left", fontsize=15,
                 fontweight="bold", color=INK)
    fig.text(0.008, 0.955,
             "Five AmeriFlux towers, freshwater marsh through mangrove. "
             "Points = daily; line = 30-day mean; band = published closure "
             "uncertainty. Shaded = 2022–23 analysis window.",
             ha="left", fontsize=10, color=INK2)

    # no legend: every panel is directly labelled, so a legend would only repeat
    # itself. Identity never rests on colour alone either way.
    fig.text(0.008, 0.018,
             "Line breaks = no data (gaps are not interpolated).",
             ha="left", fontsize=9, color=MUTED)

    fig.subplots_adjust(top=0.925, bottom=0.06, left=0.055, right=0.99)
    p = f"{OUT}/et_timeseries_everglades.png"
    fig.savefig(p, dpi=170, facecolor=SURFACE)
    print(f"wrote {p}")

    # the table view the contrast WARN obliges us to provide
    rows = []
    for sid, desc, _ in SITES:
        d = et[(et.SITE_ID == sid) & (et.ET_closed_mm.notna())]
        w = d[(d.index >= "2022-01-01") & (d.index <= "2023-12-31")]
        rows.append(dict(site=sid, ecosystem=desc, days=len(d),
                         days_2022_23=len(w),
                         first=f"{d.index.min():%Y-%m}",
                         last=f"{d.index.max():%Y-%m}",
                         ET_mean=round(d.ET_closed_mm.mean(), 2),
                         ET_p10=round(d.ET_closed_mm.quantile(.10), 2),
                         ET_p90=round(d.ET_closed_mm.quantile(.90), 2),
                         closure=d.CLOSURE.iloc[0]))
    t = pd.DataFrame(rows)
    t.to_csv(f"{OUT}/et_timeseries_table.csv", index=False)
    print()
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
