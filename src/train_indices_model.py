"""ET regression from a richer biophysical feature set + scatter of true vs pred.

Features (all from Landsat bands already extracted, footprint-weighted to the
tower per overpass date, + ERA5/gridMET):
  LAI    -ln((0.69-SAVI)/0.59)/0.91    (Anderson et al.; used in DisALEXI/TSEB)
  EVI2   2.5(NIR-RED)/(NIR+2.4 RED+1)  (2-band EVI, no blue band needed)
  SAVI   1.5(NIR-RED)/(NIR+RED+0.5)
  NDVI, NDWI(Gao, NIR-SWIR), MNDWI, LST_K
  ERA5: TA, VPD, SW_IN, WS ; gridMET ETo ; season (DOY sin/cos)

Two cross-validated scatters (both honest, different questions):
  A. leave-one-YEAR-out within site  -> gap-fill / temporal skill (usable)
  B. leave-one-TOWER-out             -> spatial scaling to an unseen site
Models compared: Random Forest, Gradient Boosting, Ridge.
"""
import os
import sys
sys.path.insert(0, "/anvil/scratch/x-jwang120/coastal-et/src")
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import gapfill_model as G

R = "/anvil/scratch/x-jwang120/coastal-et"
PIX = f"{R}/data/interim/pixels"
OUT = f"{R}/data/processed"
FIG = f"{R}/figures"
SITES = ["US-Esm", "US-TaS", "US-Skr", "US-Elm", "US-EvM"]
SATF = ["LAI", "EVI2", "SAVI", "NDVI", "NDWI", "MNDWI", "LST_K"]
METF = ["TA_ERA", "VPD_ERA", "SW_IN_ERA", "WS_ERA", "ETo_mm", "DOY_sin", "DOY_cos"]
FEATS = SATF + METF


def wavg(x, w):
    w = np.where(np.isfinite(x), w, 0.0)
    x = np.where(np.isfinite(x), x, 0.0)
    return x.dot(w) / w.sum() if w.sum() > 0 else np.nan


def build():
    """Footprint-weighted biophysical features per (site,date) + ET + met."""
    et = pd.read_parquet(f"{OUT}/daily_closed_et.parquet")
    et.index = pd.to_datetime(et.index, utc=True)
    rows = []
    for s in SITES:
        p = f"{PIX}/{s}_pixels.parquet"
        if not os.path.exists(p):
            continue
        px = pd.read_parquet(p)
        px["date"] = pd.to_datetime(px["date"], utc=True)
        # NEED the raw bands to build indices; the pixel file stored NDVI/MNDWI/LST
        # but not bands, so reconstruct the extra indices from NDVI where possible
        # and from stored columns. We stored LST_K, NDVI, MNDWI per pixel.
        # SAVI/EVI2/NDWI need bands -> approximate from NDVI is NOT ok, so we
        # recompute them below in the extractor version. Here we read what exists.
        # (extract_pixels was extended to also store band-derived indices.)
        for c in ["SAVI", "EVI2", "NDWI"]:
            if c not in px.columns:
                px[c] = np.nan
        # LAI from SAVI
        savi = px["SAVI"].clip(0, 0.685)
        px["LAI"] = (-np.log((0.69 - savi) / 0.59) / 0.91).clip(0, 6)
        # aggregate footprint-weighted per date
        agg = px.groupby("date").apply(lambda x: pd.Series(
            {c: wavg(x[c].values, x.fp_weight.values) for c in
             ["LAI", "EVI2", "SAVI", "NDVI", "NDWI", "MNDWI", "LST_K"]}))
        e = et[(et.SITE_ID == s) & (et.ET_closed_mm.notna())][["ET_closed_mm"]]
        e.index = e.index.normalize()
        m = G.merged_met(s)
        gm = f"{R}/data/interim/gridmet/{s}_gridmet.parquet"
        eto = None
        if os.path.exists(gm):
            g = pd.read_parquet(gm)
            g.index = pd.to_datetime(g.index, utc=True)
            eto = g[["ETo_mm"]] if "ETo_mm" in g else None
        perdate = e.join(m, how="left")
        if eto is not None:
            perdate = perdate.join(eto, how="left")
        perdate.index = perdate.index.normalize()
        agg.index = agg.index.normalize()
        d = agg.join(perdate, how="inner")
        doy = d.index.dayofyear
        d["DOY_sin"] = np.sin(2 * np.pi * doy / 365.25)
        d["DOY_cos"] = np.cos(2 * np.pi * doy / 365.25)
        d["SITE_ID"] = s
        d["year"] = d.index.year
        rows.append(d.reset_index().rename(columns={"index": "date"}))
    return pd.concat(rows, ignore_index=True)


def models():
    return {
        "RandomForest": lambda: RandomForestRegressor(400, min_samples_leaf=3,
                                                      random_state=42, n_jobs=-1),
        "GradBoost": lambda: GradientBoostingRegressor(n_estimators=300, max_depth=2,
                                                       random_state=42),
        "Ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10)),
    }


def oof_year(d, mk):
    """out-of-fold: leave-one-year-out within each site."""
    pr = []
    for s in SITES:
        ds = d[d.SITE_ID == s]
        for y in sorted(ds.year.unique()):
            tr = ds[ds.year != y]
            te = ds[ds.year == y]
            if len(te) < 5 or len(tr) < 20:
                continue
            m = mk()
            m.fit(tr[FEATS].values, tr.ET_closed_mm.values)
            pr.append(te.assign(pred=m.predict(te[FEATS].values)))
    return pd.concat(pr)


def oof_tower(d, mk):
    """out-of-fold: leave-one-tower-out."""
    pr = []
    for s in SITES:
        tr = d[d.SITE_ID != s]
        te = d[d.SITE_ID == s]
        if len(te) < 5:
            continue
        m = mk()
        m.fit(tr[FEATS].values, tr.ET_closed_mm.values)
        pr.append(te.assign(pred=m.predict(te[FEATS].values)))
    return pd.concat(pr)


def main():
    d = build().dropna(subset=["ET_closed_mm"] + FEATS)
    print(f"{len(d)} tower-date samples (real overpass), {d.SITE_ID.nunique()} sites")
    print("  per site:", d.SITE_ID.value_counts().to_dict(), "\n")

    print(f"{'model':<14}{'LOYO R2':>9}{'LOYO MAE':>9}{'LOTO R2':>9}{'LOTO MAE':>9}")
    print("-" * 50)
    best = None
    for name, mk in models().items():
        gy = oof_year(d, mk)
        gt = oof_tower(d, mk)
        r_y = r2_score(gy.ET_closed_mm, gy.pred)
        r_t = r2_score(gt.ET_closed_mm, gt.pred)
        print(f"{name:<14}{r_y:>9.3f}{mean_absolute_error(gy.ET_closed_mm, gy.pred):>9.2f}"
              f"{r_t:>9.3f}{mean_absolute_error(gt.ET_closed_mm, gt.pred):>9.2f}")
        if best is None or r_y > best[1]:
            best = (name, r_y, gy, gt)

    # feature importance from RF on all data
    rf = models()["RandomForest"]()
    rf.fit(d[FEATS].values, d.ET_closed_mm.values)
    imp = pd.Series(rf.feature_importances_, index=FEATS).sort_values(ascending=False)
    print("\nRF importance:", ", ".join(f"{k}={v:.2f}" for k, v in imp.head(8).items()))

    # ---- scatter figure: best model, both CV modes ----
    name, _, gy, gt = best
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COL = {"US-Elm": "#4a3aa7", "US-Esm": "#008300", "US-TaS": "#eda100",
           "US-EvM": "#1baf7a", "US-Skr": "#2a78d6"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (g, title) in zip(axes, [(gy, "Leave-one-YEAR-out (gap-fill / temporal)"),
                                     (gt, "Leave-one-TOWER-out (spatial scaling)")]):
        ax.set_facecolor("#fcfcfb")
        for s in SITES:
            gg = g[g.SITE_ID == s]
            if len(gg):
                ax.scatter(gg.ET_closed_mm, gg.pred, s=22, color=COL[s],
                           alpha=0.55, edgecolors="none", label=s)
        lim = [0, 8]
        ax.plot(lim, lim, ls="--", color="#52514e", lw=1, alpha=0.6)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        r2 = r2_score(g.ET_closed_mm, g.pred)
        rmse = np.sqrt(mean_squared_error(g.ET_closed_mm, g.pred))
        mae = mean_absolute_error(g.ET_closed_mm, g.pred)
        ax.text(0.05, 0.95, f"{title}\nR² = {r2:.2f}\nRMSE = {rmse:.2f} mm/d\n"
                f"MAE = {mae:.2f}  n = {len(g)}", transform=ax.transAxes,
                fontsize=10, va="top", color="#0b0b0b")
        ax.set_xlabel("observed ET (mm/day)", fontsize=10, color="#52514e")
        ax.set_ylabel("predicted ET (mm/day)", fontsize=10, color="#52514e")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors="#52514e", labelsize=9)
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower right", ncol=1)
    fig.suptitle(f"ET regression with LAI/EVI2/NDWI/LST + ERA5  —  best model: {name}",
                 fontsize=13, fontweight="bold", color="#0b0b0b")
    fig.subplots_adjust(top=0.90, bottom=0.11, left=0.07, right=0.98, wspace=0.22)
    fig.savefig(f"{FIG}/et_scatter_indices.png", dpi=160, facecolor="#fcfcfb")
    print(f"\nwrote {FIG}/et_scatter_indices.png")
    d.to_parquet(f"{OUT}/indices_model_table.parquet")


if __name__ == "__main__":
    main()
