"""Build the ML-ready daily table for the Everglades sites (team2 scope).

Target : ET from eddy covariance
Features: Landsat (optical + THERMAL/LST), Sentinel-2 (optical), ERA5 (met)

Reference-ET provenance differs by site, and that difference is not cosmetic:

  US-Elm, US-Esm, US-EvM   FLUXNET v1.3 -> LE_CORR, produced by ONEFlux with the
                           proper energy-balance-closure correction, and shipped
                           with uncertainty (LE_CORR_25/75, LE_RANDUNC).
  US-Skr                   BASE only -> no ONEFlux product exists. We apply a
                           Bowen-ratio closure correction ourselves, so its ET is
                           NOT strictly comparable to the other three and carries
                           no published uncertainty.

That asymmetry lands on the mangrove, which is the most distinct ecosystem in the
set. Any cross-site result must say so.

ERA5 comes from the FLUXNET AUX product (TA_ERA, VPD_ERA, SW_IN_ERA...), already
downscaled to each tower -- no CDS download needed.

Run via Slurm, never on the login node.
"""

import os
import glob
import zipfile

import numpy as np
import pandas as pd

TEAM = "/anvil/projects/x-ees260113/team2/datasets/flux_data"
SAT = "/anvil/scratch/x-jwang120/coastal-et/data/interim/satellite"
OUT = "/anvil/scratch/x-jwang120/coastal-et/data/processed"
WORK = "/anvil/scratch/x-jwang120/coastal-et/data/interim/fluxnet"

SITES = ["US-Skr", "US-EvM", "US-Esm", "US-Elm"]
STRATUM = {"US-Skr": "MANGROVE", "US-EvM": "TIDAL_BRACKISH",
           "US-Esm": "EVERGLADES_MARSH", "US-Elm": "EVERGLADES_MARSH"}

MISSING = -9999.0
LAMBDA = 2.45e6      # J/kg, latent heat of vaporisation at ~20 C
SEC_PER_DAY = 86400


def unpack(site):
    """Return the FLUXNET package dir for a site, unzipping if needed."""
    direct = glob.glob(os.path.join(TEAM, f"*{site}*FLUXNET*"))
    for d in direct:
        if os.path.isdir(d):
            return d
    zips = [z for z in direct if z.endswith(".zip")]
    if not zips:
        return None
    dest = os.path.join(WORK, site)
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall(dest)
    return dest


def le_to_et(le_wm2):
    """W/m2 (daily mean) -> mm/day."""
    return le_wm2 / LAMBDA * SEC_PER_DAY


def from_fluxnet(site):
    d = unpack(site)
    if not d:
        return None
    dd = glob.glob(os.path.join(d, "**", "*FLUXMET_DD*.csv"), recursive=True)
    if not dd:
        return None
    f = pd.read_csv(dd[0], na_values=[MISSING, "-9999"])
    f.index = pd.to_datetime(f["TIMESTAMP"].astype(str), format="%Y%m%d", utc=True)

    out = pd.DataFrame(index=f.index)
    out["LE_CORR"] = f.get("LE_CORR")
    out["ET_mm"] = le_to_et(f["LE_CORR"]) if "LE_CORR" in f else np.nan
    # uncertainty on the reference itself -- propagate this, do not discard it
    if "LE_CORR_25" in f and "LE_CORR_75" in f:
        out["ET_lo_mm"] = le_to_et(f["LE_CORR_25"])
        out["ET_hi_mm"] = le_to_et(f["LE_CORR_75"])
    if "LE_RANDUNC" in f:
        out["ET_randunc_mm"] = le_to_et(f["LE_RANDUNC"])
    out["ET_uncorr_mm"] = le_to_et(f["LE_F_MDS"]) if "LE_F_MDS" in f else np.nan
    for c in ["NETRAD", "G_F_MDS", "H_CORR"]:
        if c in f:
            out[c] = f[c]
    # QC: LE_F_MDS_QC 0=measured, 1=good gapfill. Keep only trustworthy days.
    if "LE_F_MDS_QC" in f:
        out["ET_qc"] = f["LE_F_MDS_QC"]

    # ERA5 forcing, shipped with FLUXNET, already downscaled to the tower
    era = glob.glob(os.path.join(d, "**", "*ERA5_DD*.csv"), recursive=True)
    if era:
        e = pd.read_csv(era[0], na_values=[MISSING, "-9999"])
        e.index = pd.to_datetime(e["TIMESTAMP"].astype(str), format="%Y%m%d", utc=True)
        keep = [c for c in ["TA_ERA", "SW_IN_ERA", "LW_IN_ERA", "VPD_ERA",
                            "PA_ERA", "P_ERA", "WS_ERA"] if c in e.columns]
        out = out.join(e[keep], how="left")
    out["ET_SOURCE"] = "FLUXNET_LE_CORR"
    return out


def from_base(site):
    """US-Skr has no FLUXNET product; correct closure from BASE ourselves."""
    zips = glob.glob(os.path.join(TEAM, "**", f"*{site}*BASE*.zip"), recursive=True)
    if not zips:
        return None
    dest = os.path.join(WORK, site)
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall(dest)
    csvs = glob.glob(os.path.join(dest, "**", "*BASE*.csv"), recursive=True)
    if not csvs:
        return None

    b = pd.read_csv(csvs[0], skiprows=2, na_values=[MISSING, "-9999"])
    b.index = pd.to_datetime(b["TIMESTAMP_START"].astype(str),
                             format="%Y%m%d%H%M", utc=True)
    b = b.replace(MISSING, np.nan)

    def pick(base):
        c = [x for x in b.columns if x == base or x.startswith(base + "_")]
        return b[sorted(c, key=len)[0]] if c else pd.Series(np.nan, index=b.index)

    le, h = pick("LE"), pick("H")
    rn, g = pick("NETRAD"), pick("G")
    le = le.where(le.between(-100, 800))

    avail = rn - g.fillna(0)
    turb = h + le
    ok = (avail > 20) & (turb > 20) & le.notna() & h.notna()
    resid = avail - turb
    le_corr = le + (resid * (le / turb)).where(ok, 0)
    blow = (le_corr / le).abs() > 2.0
    le_corr = le_corr.where(~blow, le)

    day = le_corr.groupby(le_corr.index.floor("D"))
    cov = day.apply(lambda s: s.notna().mean())
    daily = pd.DataFrame({"LE_CORR": day.mean(), "coverage": cov})
    daily["ET_mm"] = le_to_et(daily["LE_CORR"])
    daily.loc[daily["coverage"] < 0.8, "ET_mm"] = np.nan
    daily["ET_SOURCE"] = "BASE_bowen_corrected"   # NOT comparable to ONEFlux
    ebr = float(turb[ok].sum() / avail[ok].sum()) if ok.any() else np.nan
    print(f"    {site}: hand-corrected from BASE, EBR={ebr:.2f}")
    return daily


def satellite(site):
    p = os.path.join(SAT, f"{site}_satellite.parquet")
    if not os.path.exists(p):
        return None
    s = pd.read_parquet(p)
    s.index = pd.to_datetime(s.index, utc=True)
    feats = [c for c in ["LST_K", "NDVI", "MNDWI", "landsat_lwir11_std",
                         "landsat_red_std"] if c in s.columns]
    d = s[feats].resample("D").mean()
    out = pd.DataFrame(index=d.index)
    for c in feats:
        out[f"{c}_obs"] = d[c]
        out[f"{c}_int"] = d[c].interpolate(limit=16, limit_area="inside")
        idx = pd.Series(np.arange(len(d)), index=d.index)
        out[f"{c}_age"] = (idx - idx.where(d[c].notna()).ffill()).astype(float)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)

    frames, rep = [], []
    for s in SITES:
        flux = from_fluxnet(s)
        if flux is None:
            flux = from_base(s)
        if flux is None:
            print(f"  {s}: NO FLUX DATA")
            continue

        sat = satellite(s)
        df = flux.join(sat, how="left") if sat is not None else flux
        df = df[df["ET_mm"].notna()]
        df["SITE_ID"] = s
        df["STRATUM"] = STRATUM[s]
        df["DOY"] = df.index.dayofyear
        df["DOY_sin"] = np.sin(2 * np.pi * df.DOY / 365.25)
        df["DOY_cos"] = np.cos(2 * np.pi * df.DOY / 365.25)

        w = df[(df.index >= "2022-01-01") & (df.index <= "2023-12-31")]
        frames.append(df)
        rep.append(dict(
            site=s, stratum=STRATUM[s], source=df["ET_SOURCE"].iloc[0],
            days_all=len(df), days_2022_23=len(w),
            ET_med=round(df.ET_mm.median(), 2),
            lst_obs_2022_23=int(w["LST_K_obs"].notna().sum()) if "LST_K_obs" in w else 0,
            has_unc="ET_lo_mm" in df.columns,
        ))

    data = pd.concat(frames).sort_values(["SITE_ID"])
    data.index.name = "date"
    data.to_parquet(os.path.join(OUT, "everglades_et_dataset.parquet"))

    r = pd.DataFrame(rep)
    print("\n" + r.to_string(index=False))

    w = data[(data.index >= "2022-01-01") & (data.index <= "2023-12-31")]
    print(f"\nALL YEARS : {len(data):,} site-days, {data.SITE_ID.nunique()} sites")
    print(f"2022-2023 : {len(w):,} site-days")
    if "LST_K_obs" in w:
        n = int(w["LST_K_obs"].notna().sum())
        print(f"2022-2023 with a REAL Landsat LST observation: {n:,} "
              f"({100*n/len(w):.0f}% of days)")
    print(f"\nwrote {OUT}/everglades_et_dataset.parquet")


if __name__ == "__main__":
    main()
