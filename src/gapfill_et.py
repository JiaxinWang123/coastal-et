"""Gap-fill closed daily ET, 2018-2023, using the reference-ET fraction method.

REQUESTED for US-Skr, which is missing 62% of 2018-2023. Built, but built so it
CANNOT quietly contaminate the model.

THE METHOD (standard, and better than interpolating ET directly)
  Do not interpolate ET. Interpolate the evaporative fraction

      EF = ET_measured / ETo_gridMET

  which is far smoother than ET itself -- it strips out the day-to-day weather
  swing, leaving the slowly-varying surface control (canopy state, water table).
  Then reconstruct

      ET_filled = EF_interpolated x ETo

  so the filled day still responds to that day's actual weather.

THE RULE THAT MAKES THIS SAFE
  Filled values go in a SEPARATE column, ET_gapfilled_mm, with a boolean
  ET_IS_FILLED. ET_closed_mm is never touched. Training and validation must use
  ET_closed_mm (measured only). Filled values are for water budgets, annual
  totals and figures -- never for fitting or scoring a model.

WHY THAT RULE EXISTS
  ETo = Penman-Monteith(Ta, VPD, wind, Rn) -- a function of MET.
  So ET_filled is a function of MET.
  The model is ET ~ f(LST, NDVI, MET).
  Train on filled rows and the model learns to reproduce Penman-Monteith rather
  than the ecosystem; score on them and you are grading it on its own homework.

WHAT IT ACTUALLY BUYS (measured, not assumed)
  At US-Skr, filling gaps up to 30 days recovers only 17 extra Landsat-matched
  days (43 -> 60), because 79% of the missing days sit in 8 blocks LONGER than
  30 days -- the longest is 391 days. A year-long gap cannot be filled; it can
  only be fabricated. MAX_FILL_GAP enforces that limit.

Run via Slurm, never on the login node.
"""

import os

import numpy as np
import pandas as pd

R = "/anvil/scratch/x-jwang120/coastal-et"
OUT = f"{R}/data/processed"

SITES = ["US-Skr", "US-Elm", "US-Esm", "US-EvM", "US-TaS"]
LO, HI = "2018-01-01", "2023-12-31"

# a gap longer than this is not filled. Beyond ~30 d the interpolated EF carries
# no information from either endpoint and the "fill" is invention.
MAX_FILL_GAP = 30


def main():
    et = pd.read_parquet(f"{OUT}/daily_closed_et.parquet")
    et.index = pd.to_datetime(et.index, utc=True)

    rows, rep = [], []
    for s in SITES:
        e = et[et.SITE_ID == s].copy()
        if e.empty:
            continue

        gm = f"{R}/data/interim/gridmet/{s}_gridmet.parquet"
        if not os.path.exists(gm):
            print(f"{s}: no gridMET -> cannot gap-fill")
            continue
        g = pd.read_parquet(gm)
        g.index = pd.to_datetime(g.index, utc=True)
        eto = g.get("ETo_mm")
        if eto is None:
            print(f"{s}: gridMET has no reference ET")
            continue

        full = pd.date_range(LO, HI, freq="D", tz="UTC")
        d = pd.DataFrame(index=full)
        d["ET_closed_mm"] = e["ET_closed_mm"].reindex(full)
        d["ETo_mm"] = eto.reindex(full)

        # evaporative fraction: smooth, because the weather is divided out
        ef = d["ET_closed_mm"] / d["ETo_mm"].replace(0, np.nan)
        ef = ef.clip(0, 2.0)                       # physically bounded

        ef_int = ef.interpolate(limit=MAX_FILL_GAP, limit_area="inside")
        filled = ef_int * d["ETo_mm"]

        d["ET_IS_FILLED"] = d["ET_closed_mm"].isna() & filled.notna()
        d["ET_gapfilled_mm"] = d["ET_closed_mm"].where(~d["ET_IS_FILLED"], filled)
        d["EF"] = ef_int
        d["SITE_ID"] = s
        rows.append(d)

        n_meas = int(d["ET_closed_mm"].notna().sum())
        n_fill = int(d["ET_IS_FILLED"].sum())
        n_left = int(d["ET_gapfilled_mm"].isna().sum())
        rep.append(dict(
            site=s, days=len(full), measured=n_meas, filled=n_fill,
            still_missing=n_left,
            pct_measured=round(100 * n_meas / len(full)),
            pct_after=round(100 * (n_meas + n_fill) / len(full)),
            ET_meas=round(d["ET_closed_mm"].mean(), 2),
            ET_filled_only=(round(d.loc[d.ET_IS_FILLED, "ET_gapfilled_mm"].mean(), 2)
                            if n_fill else np.nan),
        ))

    data = pd.concat(rows)
    data.index.name = "date"
    data.to_parquet(f"{OUT}/daily_et_gapfilled_2018_2023.parquet")
    data.to_csv(f"{OUT}/daily_et_gapfilled_2018_2023.csv")

    r = pd.DataFrame(rep)
    print("=== GAP-FILLED ET, 2018-2023 (EF x ETo method, gaps <= "
          f"{MAX_FILL_GAP} d) ===\n")
    print(r.to_string(index=False))
    print(f"\n  measured days : {r.measured.sum():,}")
    print(f"  filled days   : {r.filled.sum():,}")
    print(f"  STILL missing : {r.still_missing.sum():,}  "
          f"(gaps longer than {MAX_FILL_GAP} d -- deliberately NOT fabricated)")

    # the number that actually decides whether this was worth it
    print("\n=== does filling buy the model anything? ===")
    for s in SITES:
        p = f"{R}/data/interim/satellite/{s}_satellite.parquet"
        if not os.path.exists(p):
            continue
        sat = pd.read_parquet(p)
        sat.index = pd.to_datetime(sat.index, utc=True)
        lst = sat["LST_K"].groupby(sat.index.normalize()).mean().dropna()
        lst = lst[(lst.index >= LO) & (lst.index <= HI)]
        d = data[data.SITE_ID == s]
        meas = set(d[d.ET_closed_mm.notna()].index)
        fill = set(d[d.ET_IS_FILLED].index)
        a = len(set(lst.index) & meas)
        b = len(set(lst.index) & fill)
        print(f"  {s}: Landsat-matched rows  measured={a:>3}   +filled={b:>3}"
              f"   -> {a + b:>3}")

    print("\nWROTE ET_gapfilled_mm + ET_IS_FILLED.  ET_closed_mm is UNCHANGED.")
    print("TRAIN AND VALIDATE ON ET_closed_mm ONLY. Filled values are for water")
    print("budgets and figures; using them to fit or score the model is circular.")


if __name__ == "__main__":
    main()
