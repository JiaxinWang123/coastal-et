"""Produce the deliverable: DAILY, ENERGY-BALANCE-CLOSED ET from the flux towers.

This is the target variable. Everything downstream is judged against it, so its
provenance is stated per-site rather than hidden behind one column.

CLOSURE
  Eddy covariance systematically under-measures turbulent flux: H + LE < Rn - G.
  The gap (typically 10-30%) falls disproportionately on LE, so uncorrected ET is
  biased LOW. We report the energy balance ratio

      EBR = sum(H + LE) / sum(Rn - G)

  and close the gap with the Bowen-ratio method, which redistributes the residual
  between H and LE while preserving the measured Bowen ratio B = H/LE:

      LE_closed = LE + (Rn - G - H - LE) * LE/(H + LE)

  This preserves the partitioning the tower actually observed, rather than dumping
  the whole residual into LE (the "residual method", which assumes H is perfect).

PROVENANCE (this matters, do not average over it)
  US-Elm, US-Esm, US-EvM : FLUXNET v1.3 LE_CORR -- closed by ONEFlux, the
                           community-standard pipeline, WITH published
                           uncertainty (LE_CORR_25/75, LE_RANDUNC).
  US-Skr                 : no FLUXNET product exists. Closed here from BASE by
                           the same Bowen-ratio method, but with no independent
                           uncertainty and no ONEFlux QC. NOT strictly comparable.

CONVERSION
  ET (mm/day) = LE (W/m2, daily mean) / lambda(T) * 86400 / 1000 * 1000
  lambda is temperature-dependent (~0.1% per K); ignoring it biases warm sites.

DAILY AGGREGATION
  A day needs >= 80% of its half-hours present. Summing a day that is 40% missing
  systematically under-reports ET, and no amount of downstream modelling recovers
  from a target that is quietly wrong.

Run via Slurm, never on the login node.
"""

import os
import glob
import zipfile

import numpy as np
import pandas as pd

TEAM = "/anvil/projects/x-ees260113/team2/datasets/flux_data"
FLXNET = "/anvil/scratch/x-jwang120/coastal-et/data/raw/ameriflux/fluxnet"
BASEDIR = "/anvil/scratch/x-jwang120/coastal-et/data/raw/ameriflux/base"
WORK = "/anvil/scratch/x-jwang120/coastal-et/data/interim/fluxnet"
OUT = "/anvil/scratch/x-jwang120/coastal-et/data/processed"

SITES = [l.strip() for l in open(
    "/anvil/scratch/x-jwang120/coastal-et/config/flux_sites_all.txt") if l.strip()]
MISSING = -9999.0
MIN_COVERAGE = 0.80
MAX_CLOSURE_ADJ = 2.0     # a correction that doubles LE is a bad Rn, not closure


def lambda_v(Ta_C):
    """Latent heat of vaporisation, J/kg."""
    return (2.501 - 0.00237 * np.asarray(Ta_C, dtype=float)) * 1e6


def le_to_et(le_wm2, Ta_C):
    """Daily-mean LE (W/m2) -> ET (mm/day)."""
    return le_wm2 / lambda_v(Ta_C) * 86400.0


def unzip(pattern, dest):
    z = glob.glob(pattern, recursive=True)
    if not z:
        return None
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(z[0]) as f:
        f.extractall(dest)
    return dest


# ---------------------------------------------------------------- FLUXNET path
def fluxnet(site):
    d = None
    cand = os.path.join(FLXNET, site)
    if os.path.isdir(cand) and glob.glob(os.path.join(cand, "**", "*FLUXMET_DD*.csv"),
                                         recursive=True):
        d = cand
    if d is None:
        for c in glob.glob(os.path.join(TEAM, f"*{site}*FLUXNET*")):
            if os.path.isdir(c):
                d = c
    if d is None:
        d = unzip(os.path.join(TEAM, f"*{site}*FLUXNET*.zip"),
                  os.path.join(WORK, site))
    if d is None:
        return None

    dd = glob.glob(os.path.join(d, "**", "*FLUXMET_DD*.csv"), recursive=True)
    if not dd:
        return None
    f = pd.read_csv(dd[0], na_values=[MISSING, "-9999"])
    f.index = pd.to_datetime(f["TIMESTAMP"].astype(str), format="%Y%m%d", utc=True)

    ta = f["TA_F"] if "TA_F" in f else pd.Series(25.0, index=f.index)

    out = pd.DataFrame(index=f.index)
    out["LE_open"] = f.get("LE_F_MDS")     # gap-filled, NOT closed
    out["LE_closed"] = f.get("LE_CORR")    # ONEFlux closure-corrected
    out["H_closed"] = f.get("H_CORR")
    out["NETRAD"] = f.get("NETRAD")
    out["G"] = f.get("G_F_MDS")
    out["TA_C"] = ta

    # Several sites (US-TaS, the Elkhorn cluster) have no NETRAD *column* but DO
    # measure all four radiation components. Rebuild Rn from them:
    #     Rn = SW_IN - SW_OUT + LW_IN - LW_OUT
    # This is measured, not estimated -- and crucially it uses NO satellite input,
    # so the target stays independent of the LST we later predict it from. Deriving
    # Rn from satellite LST would leak the predictor into the target.
    if out["NETRAD"].isna().all():
        comp = ["SW_IN_F", "SW_OUT", "LW_IN_F", "LW_OUT"]
        if all(c in f for c in comp):
            rn = f["SW_IN_F"] - f["SW_OUT"] + f["LW_IN_F"] - f["LW_OUT"]
            if rn.notna().sum() > 0:
                out["NETRAD"] = rn
                out["RN_SOURCE"] = "components"

    # ONEFlux only emits LE_CORR when BOTH NETRAD and G are present. 16 of our 23
    # FLUXNET sites lack G, so they get no LE_CORR at all -- even though 11 of
    # them DO have NETRAD. Close those ourselves.
    #
    # On a DAILY timescale G integrates to ~0 (heat stored by day is released at
    # night), so G=0 is a defensible daily approximation. It would NOT be at
    # half-hourly resolution, where G is a large term.
    if out["LE_closed"].notna().sum() == 0:
        # Close at HALF-HOURLY resolution, exactly as ONEFlux does, then aggregate.
        # The Bowen ratio B = H/LE varies strongly through the day, so correcting a
        # DAILY mean applies a daily-averaged B and gives a different answer. Doing
        # it on daily means would leave this site methodologically inconsistent with
        # the ONEFlux sites -- a difference sitting right on the ecological gradient.
        hh = fluxnet_halfhourly(site)
        if hh is not None:
            out["LE_closed"] = hh
            out["CLOSURE_NOTE"] = "bowen_G0_halfhourly"

    out["ET_open_mm"] = le_to_et(out["LE_open"], ta)
    out["ET_closed_mm"] = le_to_et(out["LE_closed"], ta)
    if "LE_CORR_25" in f and "LE_CORR_75" in f:
        out["ET_lo_mm"] = le_to_et(f["LE_CORR_25"], ta)
        out["ET_hi_mm"] = le_to_et(f["LE_CORR_75"], ta)
    if "LE_RANDUNC" in f:
        out["ET_randunc_mm"] = le_to_et(f["LE_RANDUNC"], ta)
    # LE_F_MDS_QC: fraction measured/good-quality gapfill on that day
    out["ET_qc"] = f.get("LE_F_MDS_QC")

    out["CLOSURE"] = ("ONEFlux_LE_CORR" if "CLOSURE_NOTE" not in out
                      else "FLUXNET_bowen_G0")
    return out


# ------------------------------------------------------------------- BASE path
def base(site):
    csvs = glob.glob(os.path.join(BASEDIR, f"*{site}_BASE*.csv"))
    if not csvs:
        d = unzip(os.path.join(TEAM, "**", f"*{site}*BASE*.zip"),
                  os.path.join(WORK, site))
        if d is None:
            return None
        csvs = glob.glob(os.path.join(d, "**", "*BASE*.csv"), recursive=True)
    if not csvs:
        return None

    b = pd.read_csv(csvs[0], skiprows=2, na_values=[MISSING, "-9999"])
    b.index = pd.to_datetime(b["TIMESTAMP_START"].astype(str),
                             format="%Y%m%d%H%M", utc=True)
    b = b.replace(MISSING, np.nan)

    def pick(name):
        c = [x for x in b.columns if x == name or x.startswith(name + "_")]
        return b[sorted(c, key=len)[0]] if c else pd.Series(np.nan, index=b.index)

    le, h = pick("LE"), pick("H")
    rn, g = pick("NETRAD"), pick("G")
    ta, us = pick("TA"), pick("USTAR")
    qc = pick("LE_SSITC_TEST")

    # QC: drop only what is EXPLICITLY flagged bad. A missing flag is not a
    # failed flag -- treating NaN as bad discards most of some sites.
    if qc.notna().any():
        le = le.where(qc.isna() | qc.isin([0, 1]))
    le = le.where(us.isna() | (us >= 0.1))          # low-turbulence filter
    le = le.where(le.between(-100, 800))            # physically impossible

    # ---- half-hourly Bowen-ratio closure ----
    avail = rn - g.fillna(0)
    turb = h + le
    ok = (avail > 20) & (turb > 20) & le.notna() & h.notna()
    resid = avail - turb
    le_c = le + (resid * (le / turb)).where(ok, 0)
    le_c = le_c.where((le_c / le).abs() <= MAX_CLOSURE_ADJ, le)

    hh = pd.DataFrame({"LE_open": le, "LE_closed": le_c, "H": h,
                       "NETRAD": rn, "G": g, "TA_C": ta,
                       "avail": avail.where(ok), "turb": turb.where(ok)})

    day = hh.groupby(hh.index.floor("D"))
    out = pd.DataFrame({
        "LE_open": day["LE_open"].mean(),
        "LE_closed": day["LE_closed"].mean(),
        "NETRAD": day["NETRAD"].mean(),
        "G": day["G"].mean(),
        "TA_C": day["TA_C"].mean(),
        "coverage": day["LE_closed"].apply(lambda s: s.notna().mean()),
    })
    out["ET_open_mm"] = le_to_et(out["LE_open"], out["TA_C"])
    out["ET_closed_mm"] = le_to_et(out["LE_closed"], out["TA_C"])
    thin = out["coverage"] < MIN_COVERAGE
    out.loc[thin, ["ET_open_mm", "ET_closed_mm"]] = np.nan
    out["CLOSURE"] = "BASE_bowen_ratio"
    return out


def fluxnet_halfhourly(site):
    """Bowen-ratio closure on the HH file, aggregated to a daily mean LE."""
    d = os.path.join(FLXNET, site)
    hh = glob.glob(os.path.join(d, "**", "*FLUXMET_HH*.csv"), recursive=True)
    if not hh:
        return None
    f = pd.read_csv(hh[0], na_values=[MISSING, "-9999"], low_memory=False)
    if "TIMESTAMP_START" not in f:
        return None
    f.index = pd.to_datetime(f["TIMESTAMP_START"].astype(str),
                             format="%Y%m%d%H%M", utc=True)

    le = f.get("LE_F_MDS")
    h = f.get("H_F_MDS")
    if le is None or h is None:
        return None

    rn = f.get("NETRAD")
    if rn is None or rn.notna().sum() == 0:
        comp = ["SW_IN_F", "SW_OUT", "LW_IN_F", "LW_OUT"]
        if not all(c in f for c in comp):
            return None
        rn = f["SW_IN_F"] - f["SW_OUT"] + f["LW_IN_F"] - f["LW_OUT"]
    g = f.get("G_F_MDS")
    g = g.fillna(0.0) if g is not None else 0.0

    turb = h + le
    avail = rn - g
    # The closure is only meaningful when there IS available energy to partition,
    # i.e. daytime. At night we KEEP the measured LE rather than nulling it --
    # nulling it would make every day fail the 80% coverage test, since a day is
    # only ~45% daylight. (ONEFlux behaves the same way.)
    ok = (avail > 20) & (turb > 20) & le.notna() & h.notna() & avail.notna()
    le_c = le + ((avail - turb) * (le / turb)).where(ok, 0)
    le_c = le_c.where((le_c / le).abs() <= MAX_CLOSURE_ADJ, le)

    day = le_c.groupby(le_c.index.floor("D"))
    daily = day.mean()
    cov = day.apply(lambda x: x.notna().mean())
    return daily.where(cov >= MIN_COVERAGE)


def ebr_of(df):
    """Energy balance ratio, from the OPEN (uncorrected) fluxes."""
    if "NETRAD" not in df or df["NETRAD"].isna().all():
        return np.nan
    avail = df["NETRAD"] - df["G"].fillna(0)
    h = df.get("H_closed", df.get("H"))
    turb = (h if h is not None else 0) + df["LE_open"]
    m = (avail > 0) & turb.notna() & avail.notna()
    return float(turb[m].sum() / avail[m].sum()) if m.any() else np.nan


import io_utils  # idempotent skip/overwrite helper (same src/ dir)


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    if not io_utils.should_write(os.path.join(OUT, "daily_closed_et.parquet"),
                                 label="daily_closed_et"):
        return   # already processed

    frames, rep = [], []
    for s in SITES:
        df = fluxnet(s)
        if df is None:
            df = base(s)
        if df is None:
            print(f"{s}: NO DATA")
            continue

        if "ET_closed_mm" not in df.columns:
            print(f"{s}: no LE_CORR/LE -- skipped")
            continue
        df = df[df["ET_closed_mm"].notna()].copy()
        if df.empty:
            # a site can have a FLUXNET package but no closed LE in our window
            print(f"{s}: zero days with closed ET -- skipped")
            continue
        df["SITE_ID"] = s
        df.index.name = "date"
        frames.append(df)

        w = df[(df.index >= "2022-01-01") & (df.index <= "2023-12-31")]
        uplift = ((df.ET_closed_mm.mean() / df.ET_open_mm.mean() - 1) * 100
                  if df.ET_open_mm.notna().any() else np.nan)
        rep.append(dict(
            site=s, closure=df.CLOSURE.iloc[0], days=len(df),
            days_22_23=len(w), EBR=round(ebr_of(df), 3),
            ET_open=round(df.ET_open_mm.mean(), 2),
            ET_closed=round(df.ET_closed_mm.mean(), 2),
            uplift_pct=round(uplift, 1),
            unc=("yes" if "ET_lo_mm" in df else "no"),
        ))

    data = pd.concat(frames).sort_index().sort_values(
        "SITE_ID", kind="stable")
    keep = [c for c in ["SITE_ID", "ET_closed_mm", "ET_open_mm", "ET_lo_mm",
                        "ET_hi_mm", "ET_randunc_mm", "ET_qc", "LE_closed",
                        "LE_open", "NETRAD", "G", "TA_C", "coverage",
                        "CLOSURE"] if c in data.columns]
    data = data[keep]
    data.to_parquet(os.path.join(OUT, "daily_closed_et.parquet"))
    data.to_csv(os.path.join(OUT, "daily_closed_et.csv"))

    r = pd.DataFrame(rep)
    print("=== DAILY ENERGY-BALANCE-CLOSED ET ===\n")
    print(r.to_string(index=False))

    w = data[(data.index >= "2022-01-01") & (data.index <= "2023-12-31")]
    print(f"\ntotal site-days : {len(data):,}")
    print(f"2022-2023       : {len(w):,}")
    print(f"\nET closed  mean : {data.ET_closed_mm.mean():.2f} mm/day  "
          f"(median {data.ET_closed_mm.median():.2f})")
    print(f"ET open    mean : {data.ET_open_mm.mean():.2f} mm/day")
    print(f"-> closure raises ET by "
          f"{100*(data.ET_closed_mm.mean()/data.ET_open_mm.mean()-1):.1f}% on average.")
    print("   That uplift is LARGER than most model-vs-model differences, which is")
    print("   why every result must be reported under both open and closed ET.")

    bad = data[(data.ET_closed_mm < -0.5) | (data.ET_closed_mm > 12)]
    print(f"\nimplausible daily values (<-0.5 or >12 mm): {len(bad)}")
    print(f"\nwrote {OUT}/daily_closed_et.parquet  (+ .csv)")


if __name__ == "__main__":
    main()
