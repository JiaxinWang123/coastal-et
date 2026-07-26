"""Production per-site ET gap-filling model + validation.

Deliverable: continuous daily ET 2018-2023 at all 5 sites, reconstructed from
satellite + meteorology, honestly validated by leave-one-year-out CV.

DESIGN (matched to what the data supports, per all prior experiments):
  * Per-site models (not spatial transfer) -- this is gap-filling a KNOWN tower,
    the task the data supports (R2~0.58) vs scaling to an unseen site (R2~0).
  * Target: ET_closed_mm, MEASURED only. Reference-ET gap-fills are never the
    target (that would be circular: ETo is a function of met).
  * Met = merge of ALL sources (team ERA5 -> FLUXNET ERA5 -> gridMET) via
    combine_first, so every ET day has meteorology from the best source present.
  * Features: LST, NDVI, MNDWI (interp within gaps, with staleness age), ECOSTRESS
    LST, and met + reference ET + season.
  * Honest skill = leave-one-YEAR-out per site. Final delivered series uses a
    model trained on ALL that site's years (more data), flagged measured vs filled.
"""
import os, sys, glob; sys.path.insert(0,"/anvil/scratch/x-jwang120/coastal-et/src")
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import train_with_ecostress as TE

R="/anvil/scratch/x-jwang120/coastal-et"; OUT=f"{R}/data/processed"
SITES=["US-Esm","US-TaS","US-Skr","US-Elm","US-EvM"]

def merged_met(site):
    """team ERA5 (clean vars) -> FLUXNET ERA5 -> gridMET, combined per-day."""
    frames=[]
    t=TE.team_era5_daily(site)          # returns None if broken (temp empty)
    if t is not None: frames.append(t[["TA_ERA","VPD_ERA","SW_IN_ERA","WS_ERA","P_ERA"]])
    f=TE.T.met_daily(site)              # FLUXNET ERA5_DD or gridMET
    if f is not None: frames.append(f)
    g=f"{R}/data/interim/gridmet/{site}_gridmet.parquet"
    if os.path.exists(g):
        gg=pd.read_parquet(g); gg.index=pd.to_datetime(gg.index,utc=True)
        ren={"TA_C":"TA_ERA","SRAD_Wm2":"SW_IN_ERA","VPD_kPa":"VPD_ERA","WS_ms":"WS_ERA","P_mm":"P_ERA"}
        frames.append(gg[[c for c in ren if c in gg]].rename(columns=ren))
    if not frames: return None
    out=frames[0]
    for fr in frames[1:]: out=out.combine_first(fr)   # first source wins, others fill
    return out

def build_site(site):
    et=pd.read_parquet(f"{OUT}/daily_closed_et.parquet"); et.index=pd.to_datetime(et.index,utc=True)
    e=et[(et.SITE_ID==site)&(et.ET_closed_mm.notna())][["ET_closed_mm"]].copy()
    e=e[(e.index>="2018-01-01")&(e.index<="2023-12-31")]
    if e.empty: return None
    # satellite optical fusion + LST
    raw=pd.read_parquet(f"{R}/data/interim/tower_point/{site}_towerpoint.parquet")
    raw.index=pd.to_datetime(raw.index,utc=True); raw=raw.groupby(raw.index.normalize()).mean()
    sat=TE.T.fuse_optical(raw,site)
    if "NDVI_fused" in sat:
        v,_=TE.T.daily_with_age(sat,e.index,"NDVI_fused",30); e["NDVI_fused"]=v.reindex(e.index)
    m=merged_met(site)
    if m is not None: e=e.join(m,how="left")
    gm=f"{R}/data/interim/gridmet/{site}_gridmet.parquet"
    if os.path.exists(gm):
        g=pd.read_parquet(gm); g.index=pd.to_datetime(g.index,utc=True)
        if "ETo_mm" in g: e=e.join(g[["ETo_mm"]],how="left")
    doy=e.index.dayofyear
    e["DOY_sin"]=np.sin(2*np.pi*doy/365.25); e["DOY_cos"]=np.cos(2*np.pi*doy/365.25)
    e["SITE_ID"]=site
    return e

# Gap-filling predicts days WITHOUT a satellite overpass, so features must be
# available EVERY day. Met + reference ET + season are daily-continuous. NDVI is
# interpolated (limit 30 d) -- dense enough to fill. Raw LST/ECOSTRESS are too
# sparse to fill from and do not add skill (all prior experiments), so excluded.
FEATS=["NDVI_fused","TA_ERA","VPD_ERA","SW_IN_ERA","WS_ERA","P_ERA",
       "ETo_mm","DOY_sin","DOY_cos"]

def rf(): return RandomForestRegressor(500,min_samples_leaf=2,random_state=42,n_jobs=-1)

def main():
    val_rows=[]; series=[]
    print(f"{'site':<8}{'n_meas':>7}{'LOYO R2':>9}{'MAE':>7}{'RMSE':>7}{'feats':>7}")
    print("-"*46)
    for s in SITES:
        d=build_site(s)
        if d is None: continue
        feats=[c for c in FEATS if c in d.columns]
        d2=d.dropna(subset=["ET_closed_mm"]+feats)
        # leave-one-year-out validation
        preds=[]
        for y in sorted(d2.index.year.unique()):
            tr=d2[d2.index.year!=y]; te=d2[d2.index.year==y]
            if len(te)<10 or len(tr)<30: continue
            m=rf(); m.fit(tr[feats].values,tr.ET_closed_mm.values)
            preds.append(te.assign(pred=m.predict(te[feats].values)))
        if preds:
            P=pd.concat(preds)
            r2=r2_score(P.ET_closed_mm,P.pred); mae=mean_absolute_error(P.ET_closed_mm,P.pred)
            rmse=np.sqrt(mean_squared_error(P.ET_closed_mm,P.pred))
            print(f"{s:<8}{len(d2):>7}{r2:>9.3f}{mae:>7.2f}{rmse:>7.2f}{len(feats):>7}")
            val_rows.append(dict(site=s,n=len(d2),R2=r2,MAE=mae,RMSE=rmse))
            P["SITE_ID"]=s; series.append(P[["SITE_ID","ET_closed_mm","pred"]])
        # final delivered continuous series: model on ALL years, predict every day with features
        full=pd.date_range("2018-01-01","2023-12-31",freq="D",tz="UTC")
        dfull=d.reindex(full)
        # interpolate features across the full axis for prediction where possible
        Xall=dfull[feats].interpolate(limit=30,limit_area="inside")
        mask=Xall.notna().all(axis=1)
        m=rf(); m.fit(d2[feats].values,d2.ET_closed_mm.values)
        out=pd.DataFrame(index=full)
        out["SITE_ID"]=s
        out["ET_measured"]=dfull["ET_closed_mm"]
        out["ET_predicted"]=np.nan
        out.loc[mask,"ET_predicted"]=m.predict(Xall[mask].values)
        # final product: measured where available, else predicted
        out["ET_final"]=out["ET_measured"].combine_first(out["ET_predicted"])
        out["is_filled"]=out["ET_measured"].isna()&out["ET_predicted"].notna()
        out.to_parquet(f"{OUT}/gapfilled_{s}.parquet")

    v=pd.DataFrame(val_rows)
    v.to_csv(f"{OUT}/gapfill_validation.csv",index=False)
    pd.concat(series).to_parquet(f"{OUT}/gapfill_loyo_predictions.parquet")
    print("-"*46)
    print(f"{'POOLED':<8}{int(v.n.sum()):>7}{r2_score(pd.concat(series).ET_closed_mm,pd.concat(series).pred):>9.3f}")
    # coverage gained
    print("\n=== coverage: measured -> after gap-fill (2018-2023) ===")
    for s in SITES:
        p=f"{OUT}/gapfilled_{s}.parquet"
        if not os.path.exists(p): continue
        o=pd.read_parquet(p)
        nm=int(o.ET_measured.notna().sum()); nf=int(o.is_filled.sum())
        print(f"  {s}: measured {nm:>4}  +filled {nf:>4}  -> {nm+nf:>4}/2191 days "
              f"({100*(nm+nf)/2191:.0f}%)")
    print(f"\nwrote gapfilled_<site>.parquet, gapfill_validation.csv")

if __name__=="__main__":
    main()
