"""Predict daily closed ET from Landsat + Sentinel-2 + ERA5. RF and MLP.

SPLIT (as specified)
  TRAIN  US-Esm, US-TaS, US-Skr   2018-2021
  TEST   US-Elm, US-EvM           2022-2023

This is a leave-two-SITES-out AND leave-time-out split at once: the test sites
are ecosystems the model has never seen, in years it has never seen. That is the
honest test of "can this scale to an unmonitored marsh", and it will score far
worse than a random split. That gap is the finding, not a failure.

BASELINES ARE NOT OPTIONAL
  An R2 means nothing on its own. We report three baselines the model must beat:
    mean      predict the training mean every day        (R2 = 0 by definition)
    ETo       predict gridMET reference ET directly      (free, no model)
    met-only  RF on meteorology alone, no satellite      (does satellite add anything?)
  If the full model cannot beat met-only, the satellite data is decorative and
  the entire premise of the project is in question. That is worth knowing early.

TARGET
  ET_closed_mm -- MEASURED, energy-balance-closed. Gap-filled days are excluded:
  gap-filled ET is a function of ETo, which is a function of met, so training or
  scoring on them would grade the model on its own inputs.

Run via Slurm, never on the login node.
"""

import os
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

R = "/anvil/scratch/x-jwang120/coastal-et"
TP = f"{R}/data/interim/tower_point"
FLXNET = f"{R}/data/raw/ameriflux/fluxnet"
OUT = f"{R}/data/processed"

TRAIN_SITES = ["US-Esm", "US-TaS", "US-Skr"]
TEST_SITES = ["US-Elm", "US-EvM"]
TRAIN_YEARS = ("2018-01-01", "2021-12-31")
TEST_YEARS = ("2022-01-01", "2023-12-31")

SEED = 42
MAX_GAP = 16          # LST staleness cap when using the filled variant

# Only the met variables available at EVERY site. US-Skr has no FLUXNET product
# and falls back to gridMET, which carries no LW_IN or PA -- requiring those
# silently dropped every single mangrove row.
MET = ["TA_ERA", "SW_IN_ERA", "VPD_ERA", "WS_ERA", "P_ERA"]
# LANDSAT AND SENTINEL-2 ARE COMPLEMENTARY, NOT CO-REQUIRED.
#   thermal  -> Landsat only. Sentinel-2 has no thermal band, full stop.
#   optical  -> BOTH. Sentinel-2 (5-day revisit) fills the gaps between Landsat
#               overpasses (16-day), roughly tripling optical coverage.
# Requiring both sensors on the same day collapsed 464 usable rows to 13.
#
# The two sensors' NDVI are NOT on the same scale (different band responses), so
# S2 is cross-calibrated to the Landsat scale on coincident days before fusing.
SAT = ["LST_K", "LST_age_d", "NDVI_fused", "MNDWI_fused", "OPT_age_d"]
OPT_MAX_GAP = 20     # optical state moves slowly; 20 d fill is defensible
LST_MAX_GAP = 16     # one Landsat cycle. Beyond that, LST is invention.
PAIR_TOL = 2         # days, for the cross-calibration

# Window is now selectable. 90 m is below the noise floor (3x3 Landsat pixels vs
# ~12 m geolocation jitter); 250 m ~ the marsh EC footprint. Set COASTAL_ET_WIN.
WINDOW = os.environ.get("COASTAL_ET_WIN", "250")
EXTRA = ["ETo_mm", "DOY_sin", "DOY_cos"]


def met_daily(site):
    import glob
    f = glob.glob(f"{FLXNET}/{site}/**/*ERA5_DD*.csv", recursive=True)
    if f:
        e = pd.read_csv(f[0], na_values=[-9999, "-9999"], low_memory=False)
        e.index = pd.to_datetime(e["TIMESTAMP"].astype(str), format="%Y%m%d", utc=True)
        return e[[c for c in MET if c in e.columns]]
    # US-Skr has no FLUXNET product -> gridMET, renamed to the same schema
    p = f"{R}/data/interim/gridmet/{site}_gridmet.parquet"
    if not os.path.exists(p):
        return None
    g = pd.read_parquet(p)
    g.index = pd.to_datetime(g.index, utc=True)
    ren = {"TA_C": "TA_ERA", "SRAD_Wm2": "SW_IN_ERA", "VPD_kPa": "VPD_ERA",
           "WS_ms": "WS_ERA", "P_mm": "P_ERA"}
    return g[[c for c in ren if c in g]].rename(columns=ren)


def build():
    et = pd.read_parquet(f"{OUT}/daily_closed_et.parquet")
    et.index = pd.to_datetime(et.index, utc=True)

    frames = []
    for s in TRAIN_SITES + TEST_SITES:
        e = et[(et.SITE_ID == s) & (et.ET_closed_mm.notna())][["ET_closed_mm"]]
        if e.empty:
            print(f"  {s}: no closed ET")
            continue

        tp = f"{TP}/{s}_towerpoint.parquet"
        if not os.path.exists(tp):
            print(f"  {s}: no tower-point satellite")
            continue
        raw = pd.read_parquet(tp)
        raw.index = pd.to_datetime(raw.index, utc=True)
        raw = raw.groupby(raw.index.normalize()).mean()
        sat = fuse_optical(raw, s)

        m = met_daily(s)
        gm = f"{R}/data/interim/gridmet/{s}_gridmet.parquet"
        eto = None
        if os.path.exists(gm):
            g = pd.read_parquet(gm)
            g.index = pd.to_datetime(g.index, utc=True)
            eto = g[["ETo_mm"]] if "ETo_mm" in g else None

        d = e.copy()
        for col, gap, agecol in [("LST_K", LST_MAX_GAP, "LST_age_d"),
                                 ("NDVI_fused", OPT_MAX_GAP, "OPT_age_d"),
                                 ("MNDWI_fused", OPT_MAX_GAP, None)]:
            if col not in sat:
                continue
            v, a = daily_with_age(sat, e.index, col, gap)
            d[col] = v.reindex(d.index)
            if agecol:
                d[agecol] = a.reindex(d.index)
        if m is not None:
            d = d.join(m, how="left")
        if eto is not None:
            d = d.join(eto, how="left")

        doy = d.index.dayofyear
        d["DOY_sin"] = np.sin(2 * np.pi * doy / 365.25)
        d["DOY_cos"] = np.cos(2 * np.pi * doy / 365.25)
        d["SITE_ID"] = s
        frames.append(d)

    return pd.concat(frames).sort_index()


def fuse_optical(raw, site):
    """Fuse Landsat + Sentinel-2 optical; keep LST from Landsat alone.

    Sentinel-2 revisits every ~5 d, Landsat every ~16 d, so S2 roughly triples
    optical coverage. But the two sensors' NDVI differ systematically (band
    responses are not identical), so a naive concat would inject a step change
    every time the source flips. We regress S2 onto Landsat on near-coincident
    days and map S2 onto the Landsat scale before filling.
    """
    out = pd.DataFrame(index=raw.index)

    for name, ls_c, s2_c in [("NDVI", f"NDVI_w{WINDOW}", f"NDVI_S2_w{WINDOW}"),
                             ("MNDWI", f"MNDWI_w{WINDOW}", f"MNDWI_S2_w{WINDOW}")]:
        ls = raw[ls_c] if ls_c in raw else pd.Series(dtype=float, index=raw.index)
        s2 = raw[s2_c] if s2_c in raw else pd.Series(dtype=float, index=raw.index)

        # Cross-calibrate S2 -> Landsat. Pair within PAIR_TOL days, not on exact
        # same-day overpasses: the two satellites almost never coincide, so an
        # exact join yields 0-20 pairs and the regression degenerates (slope 0.01,
        # i.e. "S2 tells us nothing"), which then fills 70% of the record with a
        # near-CONSTANT. A +/-2 day pairing is safe for vegetation state.
        a, b, n_pair, r = 0.0, 1.0, 0, np.nan
        L = ls.dropna().rename("l").to_frame().sort_index()
        Sm = s2.dropna().rename("s").to_frame().sort_index()
        if len(L) >= 8 and len(Sm) >= 8:
            both = pd.merge_asof(L, Sm, left_index=True, right_index=True,
                                 direction="nearest",
                                 tolerance=pd.Timedelta(days=PAIR_TOL)).dropna()
            if len(both) >= 8 and both.s.std() > 1e-3:
                b, a = np.polyfit(both.s.values, both.l.values, 1)
                r = float(np.corrcoef(both.s.values, both.l.values)[0, 1])
                n_pair = len(both)
        # a degenerate fit means S2 carries no usable information about Landsat
        # NDVI here -- do NOT fill with a constant; leave the gaps as gaps.
        if n_pair < 8 or not np.isfinite(r) or abs(r) < 0.3:
            s2_cal = pd.Series(np.nan, index=s2.index)
            if name == "NDVI":
                print(f"    {site}: S2 calibration REJECTED "
                      f"(n={n_pair}, r={r if np.isfinite(r) else float('nan'):.2f}) "
                      f"-> S2 NOT used to fill; gaps left as gaps")
        else:
            s2_cal = a + b * s2

        fused = ls.combine_first(s2_cal)      # Landsat wins; S2 fills its gaps
        out[f"{name}_fused"] = fused
        if name == "NDVI" and s2_cal.notna().any():
            print(f"    {site}: S2->Landsat NDVI  slope={b:.2f} int={a:+.2f} "
                  f"r={r:.2f} (n={n_pair} pairs within {PAIR_TOL}d)")
            print(f"      optical days: Landsat {int(ls.notna().sum()):>3}"
                  f" -> fused {int(fused.notna().sum()):>3}")

    lst_col = f"LST_K_w{WINDOW}"
    if lst_col in raw:
        out["LST_K"] = raw[lst_col]           # thermal: Landsat only, no substitute
    return out


def daily_with_age(sat, index, col, max_gap):
    """Reindex to a daily axis, interpolate within max_gap, and record staleness."""
    full = pd.date_range(index.min(), index.max(), freq="D", tz="UTC")
    s = sat[col].reindex(full)
    filled = s.interpolate(limit=max_gap, limit_area="inside")
    i = pd.Series(np.arange(len(s)), index=full)
    age = (i - i.where(s.notna()).ffill()).astype(float)
    return filled, age


def score(y, p, label):
    return dict(model=label, n=len(y),
                R2=round(r2_score(y, p), 3),
                RMSE=round(float(np.sqrt(mean_squared_error(y, p))), 3),
                MAE=round(mean_absolute_error(y, p), 3),
                bias=round(float(np.mean(p - y)), 3))


def main():
    data = build()

    feats = [c for c in SAT + MET + EXTRA if c in data.columns]
    need = ["ET_closed_mm"] + feats

    tr = data[(data.SITE_ID.isin(TRAIN_SITES)) &
              (data.index >= TRAIN_YEARS[0]) & (data.index <= TRAIN_YEARS[1])]
    te = data[(data.SITE_ID.isin(TEST_SITES)) &
              (data.index >= TEST_YEARS[0]) & (data.index <= TEST_YEARS[1])]

    tr = tr.dropna(subset=need)
    te = te.dropna(subset=need)

    print("=== SPLIT ===")
    print(f"  TRAIN {TRAIN_SITES}  {TRAIN_YEARS[0][:4]}-{TRAIN_YEARS[1][:4]}")
    for s in TRAIN_SITES:
        print(f"    {s}: {int((tr.SITE_ID == s).sum()):>4} rows")
    print(f"  TEST  {TEST_SITES}  {TEST_YEARS[0][:4]}-{TEST_YEARS[1][:4]}")
    for s in TEST_SITES:
        print(f"    {s}: {int((te.SITE_ID == s).sum()):>4} rows")
    print(f"\n  train n={len(tr)}   test n={len(te)}   features={len(feats)}")
    if len(tr) < 30 or len(te) < 20:
        raise SystemExit("too few rows after the join -- fix the data, not the model")

    Xtr, ytr = tr[feats].values, tr.ET_closed_mm.values
    Xte, yte = te[feats].values, te.ET_closed_mm.values

    rows = []

    # ---- baselines the model MUST beat -------------------------------------
    rows.append(score(yte, np.full_like(yte, ytr.mean()), "baseline: train mean"))
    if "ETo_mm" in te:
        rows.append(score(yte, te.ETo_mm.values, "baseline: gridMET ETo (no model)"))

    met_f = [c for c in MET + ["DOY_sin", "DOY_cos"] if c in feats]
    rf_met = RandomForestRegressor(n_estimators=500, min_samples_leaf=3,
                                   random_state=SEED, n_jobs=-1)
    rf_met.fit(tr[met_f].values, ytr)
    rows.append(score(yte, rf_met.predict(te[met_f].values),
                      "baseline: RF, MET ONLY (no satellite)"))

    # ---- the models --------------------------------------------------------
    rf = RandomForestRegressor(n_estimators=800, min_samples_leaf=2,
                               random_state=SEED, n_jobs=-1)
    rf.fit(Xtr, ytr)
    pred_rf = rf.predict(Xte)
    rows.append(score(yte, pred_rf, "Random Forest (all features)"))

    mlp = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu",
                     alpha=1e-2, learning_rate_init=1e-3, max_iter=3000,
                     early_stopping=True, n_iter_no_change=30,
                     random_state=SEED))
    mlp.fit(Xtr, ytr)
    pred_mlp = mlp.predict(Xte)
    rows.append(score(yte, pred_mlp, "MLP (all features)"))

    # in-sample, to expose the generalisation gap
    rows.append(score(ytr, rf.predict(Xtr), "  [RF on its own TRAINING data]"))

    res = pd.DataFrame(rows)
    print("\n=== RESULTS: predicting UNSEEN SITES in UNSEEN YEARS ===\n")
    print(res.to_string(index=False))

    print("\n=== per test site ===")
    for s in TEST_SITES:
        m = te.SITE_ID == s
        if m.sum() < 5:
            continue
        print(f"  {s} (n={int(m.sum())}):")
        print(f"     RF : {score(yte[m.values], pred_rf[m.values], '')['R2']:>6} R2   "
              f"RMSE {score(yte[m.values], pred_rf[m.values], '')['RMSE']}   "
              f"bias {score(yte[m.values], pred_rf[m.values], '')['bias']}")
        print(f"     MLP: {score(yte[m.values], pred_mlp[m.values], '')['R2']:>6} R2   "
              f"RMSE {score(yte[m.values], pred_mlp[m.values], '')['RMSE']}   "
              f"bias {score(yte[m.values], pred_mlp[m.values], '')['bias']}")

    imp = pd.Series(rf.feature_importances_, index=feats).sort_values(ascending=False)
    print("\n=== RF feature importance (top 12) ===")
    for k, v in imp.head(12).items():
        print(f"  {k:<16} {v:.3f}  {'#' * int(v * 120)}")

    res.to_csv(f"{OUT}/model_results.csv", index=False)
    te = te.assign(pred_rf=pred_rf, pred_mlp=pred_mlp)
    te.to_parquet(f"{OUT}/model_predictions.parquet")
    imp.to_csv(f"{OUT}/rf_feature_importance.csv")
    print(f"\nwrote {OUT}/model_results.csv, model_predictions.parquet")


if __name__ == "__main__":
    main()
