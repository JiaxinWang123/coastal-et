"""Instantaneous-ET approach with evaporative-fraction self-preservation.

The physically correct match: an instantaneous satellite LST at the ~10:50 local
Landsat overpass is paired with the tower's INSTANTANEOUS latent heat flux in the
same half-hour, not with a 24-hour ET total. We then:

  1. Match each overpass to the tower half-hourly LE, Rn, G in a +/-45 min window.
  2. Instantaneous evaporative fraction  EF = LE / (Rn - G)  -- the quantity LST
     actually constrains, and (by self-preservation) ~constant through the day.
  3. Train a model to predict EF from the instantaneous satellite features + met.
  4. Scale to daily:  ET_daily = EF * (Rn_daily - G_daily) / lambda   (EF self-
     preservation; the standard DisALEXI / PT-JPL / OpenET step).

Validated against the measured daily closed ET on the three CV schemes.
This removes the instantaneous-vs-daily and nocturnal-ET mismatches, and recovers
US-Skr (its BASE half-hourly file has LE + NETRAD even though the daily product
lacks net radiation).
"""
import os
import sys
import glob
sys.path.insert(0, "/anvil/scratch/x-jwang120/coastal-et/src")
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.metrics import r2_score, mean_absolute_error
import train_indices_model as T

R = "/anvil/scratch/x-jwang120/coastal-et"
FLX = f"{R}/data/raw/ameriflux/fluxnet"
SKRB = f"{R}/data/interim/fluxnet/US-Skr"
SITES = ["US-Esm", "US-TaS", "US-Skr", "US-Elm", "US-EvM"]
LAMBDA = 2.45e6
W2MM = 86400.0 / LAMBDA          # daily-mean W/m2 -> mm/day
# Landsat overpass ~15:50 UTC; take the half-hours in 15:00-16:30 UTC
OVP_LO, OVP_HI = 15, 17
SATF = ["LAI", "EVI2", "SAVI", "NDVI", "NDWI", "MNDWI", "LST_K"]


def hh(site):
    """Half-hourly flux+energy for a site (FLUXNET, or Skr BASE), UTC index."""
    f = glob.glob(f"{FLX}/{site}/**/*FLUXMET_HH*.csv", recursive=True)
    if f:
        c = pd.read_csv(f[0], na_values=[-9999, "-9999"], low_memory=False)
        c.index = pd.to_datetime(c["TIMESTAMP_START"].astype(str), format="%Y%m%d%H%M", utc=True)

        def col(*n):
            for x in n:
                if x in c:
                    return pd.to_numeric(c[x], errors="coerce")
            return pd.Series(np.nan, index=c.index)
        rn = col("NETRAD")
        if rn.notna().sum() == 0 and all(x in c for x in ["SW_IN_F", "SW_OUT", "LW_IN_F", "LW_OUT"]):
            rn = col("SW_IN_F") - col("SW_OUT") + col("LW_IN_F") - col("LW_OUT")
        return pd.DataFrame({"LE": col("LE_F_MDS", "LE"), "H": col("H_F_MDS", "H"),
                             "Rn": rn, "G": col("G_F_MDS").fillna(0)})
    b = glob.glob(f"{SKRB}/**/*BASE_HH*.csv", recursive=True)
    if b:
        c = pd.read_csv(b[0], skiprows=2, na_values=[-9999, "-9999"], low_memory=False)
        c.index = pd.to_datetime(c["TIMESTAMP_START"].astype(str), format="%Y%m%d%H%M", utc=True)

        def col(base):
            m = [x for x in c.columns if x == base or x.startswith(base + "_")]
            return pd.to_numeric(c[sorted(m, key=len)[0]], errors="coerce") if m else pd.Series(np.nan, index=c.index)
        return pd.DataFrame({"LE": col("LE"), "H": col("H"), "Rn": col("NETRAD"),
                             "G": col("G").fillna(0)})
    return None


def energy(site, verbose=False):
    """Per-date: overpass EF + DAYTIME-integrated available energy for EF scaling.

    Daily ET = EF * (daytime integral of Rn-G)/lambda. The daytime integral (not
    the 24 h MEAN Rn, which is dragged down by negative nighttime net radiation)
    is the energy available for evapotranspiration.
    """
    d = hh(site)
    if d is None:
        if verbose:
            print(f"    {site}: no half-hourly file")
        return None
    d = d[(d.index >= "2018-01-01") & (d.index <= "2023-12-31")]
    ov = d[(d.index.hour >= OVP_LO) & (d.index.hour < OVP_HI)]
    gi = ov.groupby(ov.index.normalize()).mean()
    gi["avail"] = gi.Rn - gi.G
    gi = gi[gi.avail > 50]                          # daytime overpass, real energy
    gi["EF"] = (gi.LE / gi.avail).clip(0, 1.2)
    gi["LE_inst"] = gi.LE

    # daytime-integrated available energy -> mm equivalent (Rd_day)
    ae = (d.Rn - d.G)
    day = ae.where(ae > 0, 0.0)                     # only daytime positive energy
    rd = day.groupby(day.index.normalize()).sum() * 1800.0 / LAMBDA   # J/m2 -> mm
    out = gi[["EF", "LE_inst", "avail"]].join(rd.rename("Rd_day"))
    out = out[out.Rd_day > 0.5]
    if verbose:
        print(f"    {site}: {len(out)} overpass days, EF med {out.EF.median():.2f}, "
              f"Rd_day med {out.Rd_day.median():.1f} mm")
    return out


def build():
    """Merge instantaneous EF + daily Rn with footprint-weighted satellite + daily met."""
    sat = T.build()                                # footprint-weighted per overpass + daily met + ET
    sat["date"] = pd.to_datetime(sat["date"], utc=True).dt.normalize()
    frames = []
    for s in SITES:
        e = energy(s, verbose=True)
        if e is None:
            continue
        e.index = e.index.normalize()
        ss = sat[sat.SITE_ID == s].copy()
        ss = ss.merge(e, left_on="date", right_index=True, how="inner")
        frames.append(ss)
    d = pd.concat(frames, ignore_index=True)
    # instantaneous ET (mm/day equivalent, from EF x daily energy) target check
    d["ET_daily_from_EF"] = d.EF * d.Rd_day       # EF x daytime energy (mm)
    d["year"] = d.date.dt.year
    return d


MET = ["TA_ERA", "VPD_ERA", "SW_IN_ERA", "WS_ERA", "DOY_sin", "DOY_cos"]
FEATS = SATF + MET


def gp():
    return make_pipeline(StandardScaler(), GaussianProcessRegressor(
        kernel=ConstantKernel() * RBF() + WhiteKernel(), normalize_y=True,
        alpha=1e-3, random_state=0))


def evaluate(d, target, scheme, to_daily=False):
    """Predict `target`; if to_daily, convert EF->daily ET and score vs closed ET."""
    from sklearn.base import clone
    yt, yp = [], []

    def fit_pred(tr, te):
        m = clone(gp()).fit(tr[FEATS].values, tr[target].values)
        p = m.predict(te[FEATS].values)
        if to_daily:
            p = np.clip(p, 0, 1.2) * te.Rd_day.values      # EF x daytime energy
            return p, te.ET_closed_mm.values
        return p, te[target].values

    if scheme == "random":
        for tri, tei in KFold(10, shuffle=True, random_state=0).split(d):
            p, o = fit_pred(d.iloc[tri], d.iloc[tei]); yp.append(p); yt.append(o)
    elif scheme == "year":
        for s in SITES:
            ds = d[d.SITE_ID == s]
            for y in sorted(ds.year.unique()):
                tr, te = ds[ds.year != y], ds[ds.year == y]
                if len(te) < 5 or len(tr) < 20:
                    continue
                p, o = fit_pred(tr, te); yp.append(p); yt.append(o)
    else:
        for s in SITES:
            tr, te = d[d.SITE_ID != s], d[d.SITE_ID == s]
            if len(te) < 5:
                continue
            p, o = fit_pred(tr, te); yp.append(p); yt.append(o)
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    ok = np.isfinite(yt) & np.isfinite(yp)
    return r2_score(yt[ok], yp[ok]), mean_absolute_error(yt[ok], yp[ok])


def main():
    d = build().dropna(subset=["EF", "Rd_day", "ET_closed_mm"] + FEATS).reset_index(drop=True)
    print(f"{len(d)} overpass matches, {d.SITE_ID.nunique()} sites: "
          f"{d.SITE_ID.value_counts().to_dict()}")
    print(f"  instantaneous EF: median {d.EF.median():.2f} (IQR {d.EF.quantile(.25):.2f}-{d.EF.quantile(.75):.2f})")
    # sanity: EF*daily energy vs measured daily ET
    r = r2_score(d.ET_closed_mm, d.ET_daily_from_EF)
    print(f"  EF-scaled daily ET vs measured (no ML, direct): R2={r:.3f}, "
          f"MAE={mean_absolute_error(d.ET_closed_mm, d.ET_daily_from_EF):.2f}\n")

    print("=== predict instantaneous EF, score in EF units ===")
    print(f"{'':<10}{'random-CV':>11}{'leave-year':>12}{'leave-tower':>13}")
    print(f"{'GP (EF)':<10}", end="")
    for sch in ["random", "year", "tower"]:
        print(f"{evaluate(d, 'EF', sch)[0]:>{11 if sch=='random' else 12 if sch=='year' else 13}.3f}", end="")
    print()

    print("\n=== predict EF -> scale to DAILY ET, score vs measured daily ET ===")
    print(f"{'GP':<10}", end="")
    for sch in ["random", "year", "tower"]:
        r2 = evaluate(d, "EF", sch, to_daily=True)[0]
        print(f"{r2:>{11 if sch=='random' else 12 if sch=='year' else 13}.3f}", end="")
    print("\n")
    d.to_parquet(f"{R}/data/processed/instantaneous_table.parquet")
    print(f"wrote instantaneous_table.parquet ({len(d)} rows)")


if __name__ == "__main__":
    main()
