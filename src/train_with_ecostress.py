"""Add ECOSTRESS LST + clean team ERA5 to the ET model. LOSO feature-set ladder.

Tests, incrementally, whether the new thermal/reanalysis data breaks the ceiling:
   MET                 clean ERA5 (VPD from T/Td) + ETo + season
   + Landsat           add Landsat LST / NDVI (10:30 overpass)
   + ECOSTRESS         add ECOSTRESS LST + solar hour (diurnal thermal)
Compares RF and Ridge (Ridge was the best transferable model).
"""
import os, sys, glob; sys.path.insert(0,"/anvil/scratch/x-jwang120/coastal-et/src")
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import train_et_models as T

R="/anvil/scratch/x-jwang120/coastal-et"
TEAM_ERA5="/anvil/projects/x-ees260113/team2/datasets/ERA5"
ECO="/anvil/scratch/x-jwang120/coastal-et/data/interim/ecostress"
SITES=["US-Esm","US-TaS","US-Skr","US-Elm","US-EvM"]
WIN=os.environ.get("COASTAL_ET_WIN","500")

def team_era5_daily(site):
    """Load clean hourly team ERA5 -> daily, VPD computed correctly from T/Td."""
    pats=[f"{TEAM_ERA5}/**/*{site}*.csv", f"{TEAM_ERA5}/**/*{site.lower()}*.csv",
          f"{TEAM_ERA5}/**/*{site.replace('US-','').lower()}*.csv"]
    files=[]
    for p in pats: files+=glob.glob(p, recursive=True)
    files=sorted(set(files))
    if not files: return None
    fr=[]
    for f in files:
        try: c=pd.read_csv(f)
        except: continue
        if "date" not in c: continue
        c["date"]=pd.to_datetime(c["date"], utc=True, errors="coerce")
        fr.append(c)
    if not fr: return None
    c=pd.concat(fr).dropna(subset=["date"]).drop_duplicates("date").set_index("date").sort_index()
    g=pd.DataFrame(index=c.index)
    T2=pd.to_numeric(c.get("temperature_2m"),errors="coerce")-273.15
    Td=pd.to_numeric(c.get("dewpoint_temperature_2m"),errors="coerce")-273.15
    # US-TaS / US-EvM team ERA5 downloads are broken: only the two solar-radiation
    # columns have data, temperature/dewpoint/wind are all empty. Bail so build()
    # falls back to the working FLUXNET/gridMET met.
    if T2.notna().sum() == 0:
        return None
    es=0.6108*np.exp(17.27*T2/(T2+237.3)); ea=0.6108*np.exp(17.27*Td/(Td+237.3))
    g["TA_ERA"]=T2; g["VPD_ERA"]=(es-ea).clip(lower=0)
    sw=pd.to_numeric(c.get("surface_solar_radiation_downwards_hourly"),errors="coerce")
    g["SW_IN_ERA"]=sw/3600.0                       # J/m2/h -> W/m2
    u=pd.to_numeric(c.get("u_component_of_wind_10m"),errors="coerce")
    v=pd.to_numeric(c.get("v_component_of_wind_10m"),errors="coerce")
    g["WS_ERA"]=np.sqrt(u**2+v**2)
    g["P_ERA"]=pd.to_numeric(c.get("total_precipitation_hourly"),errors="coerce")*1000
    g["SKT_ERA"]=pd.to_numeric(c.get("skin_temperature"),errors="coerce")-273.15
    g["SM_ERA"]=pd.to_numeric(c.get("volumetric_soil_water_layer_1"),errors="coerce")
    # ERA5's own potential ET (m/hr, sign convention negative) -> mm/day magnitude
    pev=pd.to_numeric(c.get("potential_evaporation_hourly"),errors="coerce")
    g["PET_ERA"]=pev.abs()*1000
    d=g.resample("D").agg({"TA_ERA":"mean","VPD_ERA":"mean","SW_IN_ERA":"mean",
        "WS_ERA":"mean","P_ERA":"sum","SKT_ERA":"mean","SM_ERA":"mean","PET_ERA":"sum"})
    return d

def eco_daily(site):
    p=f"{ECO}/{site}_ecostress.parquet"
    if not os.path.exists(p): return None
    e=pd.read_parquet(p); e.index=pd.to_datetime(e.index,utc=True)
    d=e.groupby(e.index.normalize()).mean()
    return d[["ECO_LST_K","ECO_solar_hour","ECO_LST_sd"]]

def build():
    et=pd.read_parquet(f"{R}/data/processed/daily_closed_et.parquet"); et.index=pd.to_datetime(et.index,utc=True)
    frames=[]
    for s in SITES:
        e=et[(et.SITE_ID==s)&(et.ET_closed_mm.notna())][["ET_closed_mm"]].copy()
        if e.empty: continue
        # Landsat/S2 from existing tower-point fusion
        raw=pd.read_parquet(f"{R}/data/interim/tower_point/{s}_towerpoint.parquet")
        raw.index=pd.to_datetime(raw.index,utc=True); raw=raw.groupby(raw.index.normalize()).mean()
        sat=T.fuse_optical(raw,s)
        for col,gap,age in [("LST_K",16,"LST_age_d"),("NDVI_fused",20,"OPT_age_d"),("MNDWI_fused",20,None)]:
            if col in sat:
                v,a=T.daily_with_age(sat,e.index,col,gap); e[col]=v.reindex(e.index)
                if age: e[age]=a.reindex(e.index)
        # ECOSTRESS
        ed=eco_daily(s)
        if ed is not None:
            full=pd.date_range(e.index.min(),e.index.max(),freq="D",tz="UTC")
            ei=ed.reindex(full)
            for c in ["ECO_LST_K","ECO_solar_hour","ECO_LST_sd"]:
                e[c]=ei[c].interpolate(limit=16,limit_area="inside").reindex(e.index)
            i=pd.Series(np.arange(len(full)),index=full)
            eage=(i-i.where(ei["ECO_LST_K"].notna()).ffill())
            e["ECO_age_d"]=eage.reindex(e.index).values
        # clean team ERA5 (fallback to FLUXNET/gridMET via train_et_models if absent)
        m=team_era5_daily(s)
        if m is None:
            m=T.met_daily(s)
        e=e.join(m,how="left")
        gm=f"{R}/data/interim/gridmet/{s}_gridmet.parquet"
        if os.path.exists(gm):
            g=pd.read_parquet(gm); g.index=pd.to_datetime(g.index,utc=True)
            if "ETo_mm" in g: e=e.join(g[["ETo_mm"]],how="left")
        doy=e.index.dayofyear
        e["DOY_sin"]=np.sin(2*np.pi*doy/365.25); e["DOY_cos"]=np.cos(2*np.pi*doy/365.25)
        e["SITE_ID"]=s; frames.append(e)
    return pd.concat(frames).sort_index()

data=build()
def agg(df):
    out=[]
    for s,g in df.groupby("SITE_ID"):
        r=g.resample("MS"); m=r.mean(numeric_only=True); c=r["ET_closed_mm"].count()
        m=m[c>=12]; m["SITE_ID"]=s; out.append(m)
    return pd.concat(out)

MET=[c for c in ["TA_ERA","VPD_ERA","SW_IN_ERA","WS_ERA","P_ERA","SKT_ERA","SM_ERA","PET_ERA","ETo_mm","DOY_sin","DOY_cos"] if c in data.columns]
LSAT=[c for c in ["LST_K","LST_age_d","NDVI_fused","MNDWI_fused","OPT_age_d"] if c in data.columns]
ECOF=[c for c in ["ECO_LST_K","ECO_solar_hour","ECO_LST_sd","ECO_age_d"] if c in data.columns]
def rf(): return RandomForestRegressor(400,min_samples_leaf=2,random_state=42,n_jobs=-1)
def rg(): return make_pipeline(StandardScaler(),Ridge(alpha=10))
def loso(df,feats,mk,drop_tas=True):
    sites=[x for x in SITES if not(drop_tas and x=="US-TaS")]
    df=df.dropna(subset=["ET_closed_mm"]+feats); P=[]
    for s in sites:
        tr=df[df.SITE_ID!=s]; te=df[df.SITE_ID==s]
        if len(te)<6 or len(tr)<20: continue
        m=mk(); m.fit(tr[feats].values,tr.ET_closed_mm.values)
        P.append(te.assign(p=m.predict(te[feats].values)))
    if not P: return np.nan,np.nan,0
    P=pd.concat(P); return r2_score(P.ET_closed_mm,P.p),mean_absolute_error(P.ET_closed_mm,P.p),len(P)

mon=agg(data)
print(f"WINDOW={WIN}  |  ECOSTRESS feats: {ECOF}")
print(f"data: {len(data)} daily rows, {len(mon)} monthly rows, {data.SITE_ID.nunique()} sites")
print(f"  rows with ECO_LST: {int(data['ECO_LST_K'].notna().sum()) if 'ECO_LST_K' in data else 0}\n")
print(f"{'feature set':<28}{'model':<8}{'R2':>8}{'MAE':>7}{'n':>5}")
print("-"*56)
for tl,agg_df in [("MONTHLY (drop TaS)",mon)]:
    print(f"# {tl}")
    for name,fs in [("MET only",MET),("MET+Landsat",MET+LSAT),
                    ("MET+Landsat+ECOSTRESS",MET+LSAT+ECOF),
                    ("MET+ECOSTRESS (no Landsat)",MET+ECOF)]:
        for mn,mk in [("RF",rf),("Ridge",rg)]:
            r,mae,n=loso(agg_df,fs,mk); print(f"{name:<28}{mn:<8}{r:>8.3f}{mae:>7.2f}{n:>5}")
        print()
