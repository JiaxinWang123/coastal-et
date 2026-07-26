"""Closed daily ET time series for all 13 modelling sites (extends the 5-site Everglades
figure to the full coastal-wetland network).

Small multiples, one panel per site, ordered west->east (Pacific delta -> Gulf ->
Atlantic). Points = daily; line = gap-aware 30-day mean (breaks at data gaps, never
bridges multi-year outages); band = published closure uncertainty where available;
shaded = 2022-23 analysis window. Every panel is directly labelled.
"""
import os
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

R = "/anvil/scratch/x-jwang120/coastal-et"
OUT = f"{R}/figures"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8a85"

IGBP = {"WET": "wetland", "EBF": "evergreen broadleaf", "CSH": "shrub",
        "GRA": "grassland", "CRO": "cropland", "SAV": "savanna", "WSA": "woody savanna",
        "OSH": "open shrub"}

# Everglades sites are absent from core_coastal_sites.csv -> supply their labels
FALLBACK = {
    "US-Elm": ("Everglades freshwater marsh (long hydroperiod)", "FL", "wetland"),
    "US-Esm": ("Everglades freshwater marsh (short hydroperiod)", "FL", "wetland"),
    "US-Skr": ("Shark River mangrove forest", "FL", "evergreen broadleaf"),
}


def main():
    os.makedirs(OUT, exist_ok=True)
    et = pd.read_parquet(f"{R}/data/processed/daily_closed_et.parquet")
    et.index = pd.to_datetime(et.index, utc=True)
    tab = pd.read_parquet(f"{R}/data/processed/more_sites_table.parquet")
    sites = sorted(tab.SITE_ID.unique())
    meta = pd.read_csv(f"{R}/data/processed/core_coastal_sites.csv").set_index("SITE_ID")

    # order west -> east by longitude
    order = sorted(sites, key=lambda s: meta.LON.get(s, 0))
    cmap = plt.cm.tab20(np.linspace(0, 1, 20))
    colours = {s: cmap[i % 20] for i, s in enumerate(order)}

    ncol, nrow = 2, 7
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 15), sharex=True, sharey=True,
                             gridspec_kw=dict(hspace=0.38, wspace=0.10))
    fig.patch.set_facecolor(SURFACE)
    axes = axes.ravel()
    x0, x1 = pd.Timestamp("2008-01-01", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC")

    for ax, sid in zip(axes, order):
        ax.set_facecolor(SURFACE)
        col = colours[sid]
        d = et[(et.SITE_ID == sid) & (et.ET_closed_mm.notna())].sort_index()
        ax.axvspan(pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC"),
                   color="#000000", alpha=0.04, lw=0, zorder=0)
        ax.plot(d.index, d.ET_closed_mm, lw=0, marker=".", ms=1.1, color=col, alpha=0.25, zorder=2)

        daily = d.ET_closed_mm.reindex(pd.date_range(d.index.min(), d.index.max(), freq="D", tz="UTC"))
        roll = daily.rolling(30, min_periods=10).mean()
        gap = daily.isna().rolling(30, min_periods=1).sum() >= 25
        roll = roll.mask(gap)
        ax.plot(roll.index, roll.values, lw=1.7, color=col, zorder=3)
        if {"ET_lo_mm", "ET_hi_mm"} <= set(d.columns) and d.ET_lo_mm.notna().any():
            lo = d.ET_lo_mm.rolling(30, min_periods=10).mean()
            hi = d.ET_hi_mm.rolling(30, min_periods=10).mean()
            ax.fill_between(d.index, lo, hi, color=col, alpha=0.13, lw=0, zorder=1)

        st = meta.STATE.get(sid, ""); nm = str(meta.SITE_NAME.get(sid, ""))[:36]
        eco = IGBP.get(meta.IGBP.get(sid, ""), meta.IGBP.get(sid, ""))
        if sid in FALLBACK and not nm.strip():
            nm, st, eco = FALLBACK[sid]
        ax.text(0.008, 0.955, sid, transform=ax.transAxes, fontsize=11.5, fontweight="bold",
                color=INK, va="top")
        ax.text(0.008, 0.85, f"{nm} · {st} · {eco}".strip(" ·"), transform=ax.transAxes,
                fontsize=8.5, color=INK2, va="top")
        ax.text(0.992, 0.93, f"{len(d):,} d   {d.ET_closed_mm.mean():.2f} mm/d",
                transform=ax.transAxes, fontsize=8.5, color=MUTED, va="top", ha="right")

        ax.set_ylim(0, 9); ax.set_yticks([0, 3, 6, 9]); ax.set_xlim(x0, x1)
        ax.grid(axis="y", color="#e6e6e2", lw=0.7, zorder=0); ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color("#d8d8d4")
        ax.tick_params(colors=INK2, labelsize=8, length=0)

    for ax in axes[len(order):]:      # hide unused panels
        ax.set_visible(False)

    fig.suptitle("Energy-balance-closed daily ET across the 13-site coastal-wetland network",
                 x=0.008, y=0.99, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.008, 0.967,
             "AmeriFlux/FLUXNET towers, ordered west→east (Pacific delta → Gulf → Atlantic). "
             "Points = daily; line = gap-aware 30-day mean; band = closure uncertainty; "
             "shaded = 2022–23 window.", ha="left", fontsize=9.5, color=INK2)
    fig.text(0.008, 0.012, "Line breaks = no data (gaps are not interpolated).",
             ha="left", fontsize=9, color=MUTED)
    fig.supylabel("Closed daily ET  (mm day$^{-1}$)", fontsize=11, color=INK2, x=0.02)
    fig.subplots_adjust(top=0.945, bottom=0.045, left=0.06, right=0.99)

    p = f"{OUT}/et_timeseries_13sites.png"
    fig.savefig(p, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {p}")

    rows = []
    for sid in order:
        d = et[(et.SITE_ID == sid) & (et.ET_closed_mm.notna())]
        w = d[(d.index >= "2022-01-01") & (d.index <= "2023-12-31")]
        rows.append(dict(site=sid, name=str(meta.SITE_NAME.get(sid, "")), state=meta.STATE.get(sid, ""),
                         IGBP=meta.IGBP.get(sid, ""), days=len(d), days_2022_23=len(w),
                         first=f"{d.index.min():%Y-%m}", last=f"{d.index.max():%Y-%m}",
                         ET_mean=round(d.ET_closed_mm.mean(), 2),
                         ET_p10=round(d.ET_closed_mm.quantile(.10), 2),
                         ET_p90=round(d.ET_closed_mm.quantile(.90), 2)))
    t = pd.DataFrame(rows)
    t.to_csv(f"{OUT}/et_timeseries_13sites_table.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
