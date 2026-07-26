"""Turn AmeriFlux BASE half-hourly data into the reference ET series.

This defines the "ground truth", so its choices matter more than any model
downstream. It is NOT ground truth in the lysimeter sense: eddy covariance is an
indirect, assumption-laden measurement. What we produce is:

    closure-corrected, QC-filtered, eddy-covariance latent heat flux, converted
    to ET, with the energy-balance residual reported so the correction's size is
    always visible.

Chain:
  raw BASE LE  ->  QC filter (SSITC flags, u*)  ->  energy-balance closure
  correction (Bowen-ratio)  ->  LE to ET via temperature-dependent lambda
  ->  daily totals with a coverage threshold

Both the corrected and uncorrected series are written. The closure correction can
move ET by 10-30%, which is larger than the difference between competing models,
so any result must be reported under both.

Usage:  python process_flux.py [SITE_ID ...]   (default: all downloaded)
Run via Slurm, never on the login node.
"""

import os
import re
import glob
import sys

import numpy as np
import pandas as pd

BASE = "/anvil/scratch/x-jwang120/coastal-et/data/raw/ameriflux/base"
OUT = "/anvil/scratch/x-jwang120/coastal-et/data/interim/flux"
SITES = "/anvil/scratch/x-jwang120/coastal-et/config/coastal_sites.csv"

MISSING = -9999.0
USTAR_MIN = 0.1          # m/s; below this, turbulence is too weak to trust fluxes
DAILY_COVERAGE = 0.80    # a day needs >=80% of half-hours to yield a daily total
MAX_CLOSURE_ADJ = 2.0    # refuse corrections that would more than double LE


def pick(cols, base):
    """AmeriFlux qualifies variable names (LE_1_1_1, LE_PI_F, TA_1_1_1...).

    Prefer the bare name, then the gap-filled PI version, then the lowest
    positional index -- which by AmeriFlux convention is the primary sensor at
    the most representative height.
    """
    cands = [c for c in cols if c == base or re.match(rf"^{base}(_PI)?(_F)?(_\d+)*$", c)]
    if not cands:
        return None
    for c in cands:
        if c == base:
            return c
    return sorted(cands, key=lambda c: (len(c), c))[0]


def latent_heat(Ta_C):
    """Latent heat of vaporisation, J/kg. Varies ~1% per 10 C; not negligible."""
    return (2.501 - 0.00237 * Ta_C) * 1e6


def load_site(path):
    df = pd.read_csv(path, skiprows=2, na_values=[MISSING, "-9999", "-9999.0"])
    ts = df["TIMESTAMP_START"].astype(str)
    df.index = pd.to_datetime(ts, format="%Y%m%d%H%M", utc=True)
    df = df.replace(MISSING, np.nan)
    return df


def process(site_id, path):
    df = load_site(path)
    cols = df.columns

    c_le = pick(cols, "LE")
    c_h = pick(cols, "H")
    if c_le is None:
        return None, f"{site_id}: no LE variable"

    c_rn = pick(cols, "NETRAD")
    c_g = pick(cols, "G")
    c_ta = pick(cols, "TA")
    c_us = pick(cols, "USTAR")
    c_qc = pick(cols, "LE_SSITC_TEST")
    c_wtd = pick(cols, "WTD")

    out = pd.DataFrame(index=df.index)
    out["LE_raw"] = df[c_le]
    out["H"] = df[c_h] if c_h else np.nan
    out["NETRAD"] = df[c_rn] if c_rn else np.nan
    out["G"] = df[c_g] if c_g else np.nan
    out["TA"] = df[c_ta] if c_ta else np.nan
    out["USTAR"] = df[c_us] if c_us else np.nan
    out["WTD"] = df[c_wtd] if c_wtd else np.nan

    n0 = out["LE_raw"].notna().sum()

    # ---- QC -------------------------------------------------------------
    le = out["LE_raw"].copy()
    if c_qc:
        # SSITC: 0 best, 1 acceptable, 2 poor (Foken scheme, collapsed).
        # A MISSING flag is not a FAILED flag. Some sites (e.g. US-Dmg) report
        # SSITC on under 1% of rows; treating NaN as "bad" discards 99.8% of a
        # perfectly good record. Drop only what is explicitly flagged poor.
        flag = pd.to_numeric(df[c_qc], errors="coerce")
        le = le.where(flag.isna() | flag.isin([0, 1]))
    n_qc = le.notna().sum()

    if c_us:
        # likewise: an unknown u* cannot be used to condemn a half-hour
        u = out["USTAR"]
        le = le.where(u.isna() | (u >= USTAR_MIN))
    n_us = le.notna().sum()

    # physically impossible LE; open-path IRGAs in salty coastal air spike badly
    le = le.where(le.between(-100, 800))
    out["LE_qc"] = le
    n_ok = le.notna().sum()

    # ---- energy-balance closure ----------------------------------------
    avail = out["NETRAD"] - out["G"].fillna(0)   # G is often unmeasured in marsh
    turb = out["H"] + out["LE_qc"]
    resid = avail - turb

    # Bowen-ratio correction: distribute the residual between H and LE while
    # preserving the observed Bowen ratio. Only where the energy balance is
    # physically sensible -- daytime, positive available energy, positive fluxes.
    ok = (avail > 20) & (turb > 20) & out["LE_qc"].notna() & out["H"].notna()
    frac = (out["LE_qc"] / turb).where(ok)
    le_corr = out["LE_qc"] + (resid * frac).where(ok, 0)

    # a correction that more than doubles LE is not a closure correction, it is
    # a bad Rn or G measurement -- fall back to uncorrected there
    blowup = (le_corr / out["LE_qc"]).abs() > MAX_CLOSURE_ADJ
    le_corr = le_corr.where(~blowup, out["LE_qc"])
    out["LE_corr"] = le_corr

    ebr = (turb.where(ok).sum() / avail.where(ok).sum()) if ok.any() else np.nan

    # ---- LE -> ET -------------------------------------------------------
    lam = latent_heat(out["TA"].fillna(out["TA"].median()))
    for src, dst in [("LE_qc", "ET_raw_mm"), ("LE_corr", "ET_corr_mm")]:
        # W/m2 -> mm per 30 min:  (J/s/m2) * 1800 s / (J/kg) / (1000 kg/m3) * 1000 mm/m
        out[dst] = out[src] / lam * 1800.0

    # ---- daily ----------------------------------------------------------
    day = out.index.floor("D")
    grp = out.groupby(day)
    cov = grp["ET_corr_mm"].apply(lambda s: s.notna().mean())
    daily = pd.DataFrame({
        "ET_corr_mm": grp["ET_corr_mm"].sum(min_count=1),
        "ET_raw_mm": grp["ET_raw_mm"].sum(min_count=1),
        "LE_corr_Wm2": grp["LE_corr"].mean(),
        "NETRAD_Wm2": grp["NETRAD"].mean(),
        "TA_C": grp["TA"].mean(),
        "WTD_m": grp["WTD"].mean(),
        "coverage": cov,
    })
    # a day missing a quarter of its half-hours cannot give a trustworthy total;
    # summing what is there would systematically under-report ET
    daily.loc[daily["coverage"] < DAILY_COVERAGE, ["ET_corr_mm", "ET_raw_mm"]] = np.nan

    stats = dict(
        site=site_id, n_raw=int(n0), n_after_qc=int(n_qc), n_after_ustar=int(n_us),
        n_final=int(n_ok), pct_kept=round(100 * n_ok / n0, 1) if n0 else 0.0,
        EBR=round(float(ebr), 3) if pd.notna(ebr) else np.nan,
        n_days=int(daily["ET_corr_mm"].notna().sum()),
        ET_median=round(float(daily["ET_corr_mm"].median()), 2),
        ET_p99=round(float(daily["ET_corr_mm"].quantile(0.99)), 2),
        has_WTD=bool(out["WTD"].notna().any()),
        has_NETRAD=bool(out["NETRAD"].notna().any()),
    )
    return (out, daily, stats), None


import io_utils  # idempotent skip/overwrite helper (same src/ dir)


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(BASE, "*BASE*.csv")))
    want = set(io_utils.clean_argv()[1:]) or None

    rows, problems = [], []
    for f in files:
        site = os.path.basename(f).split("_")[1]
        if want and site not in want:
            continue
        if not io_utils.should_write(os.path.join(OUT, f"{site}_daily.parquet"),
                                     label=f"processed flux for {site}"):
            continue   # already processed
        try:
            res, err = process(site, f)
        except Exception as e:
            problems.append(f"{site}: {type(e).__name__}: {e}")
            continue
        if err:
            problems.append(err)
            continue
        hh, daily, stats = res
        hh.to_parquet(os.path.join(OUT, f"{site}_halfhourly.parquet"))
        daily.to_parquet(os.path.join(OUT, f"{site}_daily.parquet"))
        rows.append(stats)

    s = pd.DataFrame(rows).sort_values("site")
    pd.set_option("display.width", 200)
    print(s.to_string(index=False))
    s.to_csv(os.path.join(OUT, "flux_processing_summary.csv"), index=False)

    print(f"\nsites processed: {len(s)}")
    print(f"total site-days of reference ET: {int(s.n_days.sum()):,}")
    print(f"median half-hours retained after QC: {s.pct_kept.median():.0f}%")

    # Energy-balance ratio is the headline diagnostic. Typical EC underclosure is
    # EBR 0.7-0.9; values far outside that mean Rn or G is suspect at that site.
    ebr = s.EBR.dropna()
    print(f"\nenergy balance ratio (H+LE)/(Rn-G): median={ebr.median():.2f}  "
          f"range=[{ebr.min():.2f}, {ebr.max():.2f}]")
    odd = s[(s.EBR < 0.5) | (s.EBR > 1.2)]
    if len(odd):
        print("SITES WITH SUSPECT ENERGY BALANCE (check Rn/G before trusting):")
        print(odd[["site", "EBR", "has_NETRAD"]].to_string(index=False))
    no_rn = s[~s.has_NETRAD]
    if len(no_rn):
        print(f"\nNO NET RADIATION -> no closure correction possible ({len(no_rn)}): "
              f"{', '.join(no_rn.site)}")
    print(f"\nsites with WTD (water table): {int(s.has_WTD.sum())}/{len(s)}")

    if problems:
        print(f"\nPROBLEMS ({len(problems)}):")
        for p in problems:
            print("  " + p)


if __name__ == "__main__":
    main()
