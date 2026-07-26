"""Kljun et al. (2015) 2D flux-footprint climatology per tower.

Uses the `fluxfootprints` package (Kljun 2015, Eq 14 & 17). Produces a per-tower
footprint-weight raster on a local grid, which tells us which satellite pixels
the tower's ET actually integrates, and with what weight.

INPUT AVAILABILITY / ASSUMPTIONS (documented honestly):
  u* (USTAR), WD, WS      : measured, in FLUXNET HH               [OK]
  L (Obukhov length)      : DERIVED  L = -u*^3 / (k (g/T) (H/(rho cp)))
  sigma_v                 : PARAMETERISED  sigma_v ~ 1.9 u*  (surface-layer
                            similarity; affects crosswind spread, not the
                            along-wind source distance which zm,u*,L,h set)
  zm (measurement height) : LITERATURE values (not in the FLUXNET download):
                              US-Skr  27 m (SRS-6 mangrove tower, Barr et al.)
                              marshes  4 m
  canopy height           : LITERATURE (Skr 19 m; marshes ~1 m)
  h (boundary-layer)       : 1000 m convective default (footprint weakly
                            sensitive to h for these low towers)

These assumptions are refinable if the full BADM (GRP_VAR_INFO heights, V_SIGMA)
is obtained; the source-area geometry is dominated by zm and u*, which are solid.
"""
import os, sys, glob
import numpy as np, pandas as pd

R="/anvil/scratch/x-jwang120/coastal-et"
FLX=f"{R}/data/raw/ameriflux/fluxnet"
OUT=f"{R}/data/interim/footprint"

# zm (EC height), hc (canopy height), in metres -- literature/estimated.
# Everglades from Barr et al. / FCE-LTER; the 8 added sites estimated from site type
# (AmeriFlux BADM carries no EC height; canopy for Myb/Tw4 is from their BIF HEIGHTC).
SITE_GEOM={
 "US-Skr":dict(zm=27.0, hc=19.0),   # Shark River mangrove, SRS-6
 "US-Esm":dict(zm=4.0,  hc=1.0),    # short-hydroperiod marl marsh
 "US-Elm":dict(zm=4.0,  hc=1.5),    # long-hydroperiod sawgrass
 "US-TaS":dict(zm=4.0,  hc=1.0),    # Taylor Slough marsh
 "US-EvM":dict(zm=4.0,  hc=1.0),    # saltwater-intrusion marsh
 "US-EDN":dict(zm=3.0,  hc=0.5),    # Eden Landing, CA salt-marsh restoration (Salicornia)
 "US-Myb":dict(zm=5.0,  hc=2.7),    # Mayberry, CA restored delta wetland (tall tules; BIF hc~2.7)
 "US-Tw4":dict(zm=5.0,  hc=3.0),    # Twitchell East End, CA restored wetland (tules; BIF hc~3.2)
 "US-LA3":dict(zm=3.5,  hc=0.6),    # Barataria Bay, LA Spartina saline marsh
 "US-HB1":dict(zm=3.0,  hc=0.5),    # North Inlet Crab Haul, SC Spartina salt marsh
 "US-HB4":dict(zm=3.0,  hc=0.6),    # Minim Creek, SC brackish impoundment
 "US-NC4":dict(zm=5.0,  hc=1.5),    # Alligator River, NC pocosin shrub wetland
 "US-StJ":dict(zm=3.5,  hc=0.5),    # St Jones Reserve, DE Spartina salt marsh
}
K=0.4; G=9.81; CP=1004.0; DOMAIN=600.0; DX=10.0   # 600 m half-domain, 10 m grid

def load_base(site):
    f=glob.glob(f"{R}/data/interim/fluxnet/{site}/**/*BASE_HH*.csv",recursive=True)
    if not f: return None
    c=pd.read_csv(f[0],skiprows=2,na_values=[-9999,"-9999"],low_memory=False)
    c.index=pd.to_datetime(c["TIMESTAMP_START"].astype(str),format="%Y%m%d%H%M",utc=True)
    c=c.replace(-9999,np.nan)
    def col(*names):
        for n in names:
            m=[x for x in c.columns if x==n or x.startswith(n+"_")]
            if m: return pd.to_numeric(c[sorted(m,key=len)[0]],errors="coerce")
        return pd.Series(np.nan,index=c.index)
    d=pd.DataFrame(index=c.index)
    d["ustar"]=col("USTAR"); d["wd"]=col("WD"); d["ws"]=col("WS")
    d["H"]=col("H"); d["TA"]=col("TA")+273.15; d["PA"]=col("PA")*1000.0
    return d

def load_hh(site):
    f=glob.glob(f"{FLX}/{site}/**/*FLUXMET_HH*.csv",recursive=True)
    if not f: return load_base(site)
    c=pd.read_csv(f[0],na_values=[-9999,"-9999"],low_memory=False)
    c.index=pd.to_datetime(c["TIMESTAMP_START"].astype(str),format="%Y%m%d%H%M",utc=True)
    def col(*names):
        for n in names:
            if n in c: return pd.to_numeric(c[n],errors="coerce")
        return pd.Series(np.nan,index=c.index)
    d=pd.DataFrame(index=c.index)
    d["ustar"]=col("USTAR"); d["wd"]=col("WD"); d["ws"]=col("WS","WS_F")
    d["H"]=col("H_F_MDS","H","H_CORR"); d["TA"]=col("TA_F","TA")+273.15
    d["PA"]=col("PA_F","PA")*1000.0     # kPa->Pa
    return d

def derive_L_sigmav(d):
    rho=d["PA"]/(287.05*d["TA"])                       # air density
    # Obukhov length from surface-layer definition
    L=-(d["ustar"]**3)/(K*(G/d["TA"])*(d["H"]/(rho*CP)))
    L=L.replace([np.inf,-np.inf],np.nan)
    L=L.clip(-1e4,1e4)
    sigmav=1.9*d["ustar"]                              # surface-layer similarity
    return L, sigmav

def main():
    from fluxfootprints import FFPModel
    os.makedirs(OUT,exist_ok=True)
    site=sys.argv[1]; g=SITE_GEOM[site]
    d=load_hh(site)
    if d is None:
        print(f"{site}: no HH data"); return
    d=d[(d.index>="2018-01-01")&(d.index<="2023-12-31")]
    L,sv=derive_L_sigmav(d)
    # column names expected by the package: wind_dir, umean, ust, ol, sigmav
    df=pd.DataFrame({"ustar":d.ustar,"sigmav":sv,"ol":L,"wind_dir":d.wd,"umean":d.ws}).dropna()
    df=df[(df.ustar>0.1)&(df.umean>0)&(df.wind_dir.between(0,360))]
    df=df[(g["zm"]/df.ol>-15.5)]
    print(f"{site}: {len(df)} valid half-hours  zm={g['zm']}m  "
          f"med u*={df.ustar.median():.2f}  med L={df.ol.median():.0f}",flush=True)
    if len(df)<50:
        print("  too few valid half-hours"); return
    # a footprint CLIMATOLOGY converges with a few thousand half-hours; using all
    # 57k builds 57k rasters and OOMs. Deterministic even-stride subsample keeps
    # the wind/stability distribution while bounding memory.
    CAP=6000
    if len(df)>CAP:
        df=df.iloc[::max(1,len(df)//CAP)].iloc[:CAP]
        print(f"  subsampled to {len(df)} half-hours (climatology converges well below this)")
    m=FFPModel(df, domain=[-DOMAIN,DOMAIN,-DOMAIN,DOMAIN], dx=DX, dy=DX,
               inst_height=g["zm"], crop_height=g["hc"], atm_bound_height=1000.0,
               rs=[0.5,0.8], smooth_data=True, verbosity=0)
    out=m.run() if hasattr(m,"run") else m.calc_footprint_climatology()
    # out is an xarray Dataset with variable "footprint_climatology", coords x,y
    da = out["footprint_climatology"] if "footprint_climatology" in out else out["footprint_2d"]
    fclim=np.asarray(da.values); xs=np.asarray(out["x"].values); ys=np.asarray(out["y"].values)
    fclim=np.nan_to_num(fclim); 
    if fclim.sum()>0: fclim=fclim/fclim.sum()     # normalise to sum 1
    np.savez(f"{OUT}/{site}_ffp.npz", fclim=fclim, x=np.asarray(xs), y=np.asarray(ys),
             zm=g["zm"], n_halfhours=len(df))
    # peak & 80% source-area extent
    flat=np.sort(fclim[np.isfinite(fclim)])[::-1]; cum=np.cumsum(flat)
    r80=flat[np.searchsorted(cum,0.8)] if len(flat) else np.nan
    area80=int((fclim>=r80).sum())*DX*DX/1e4    # ha
    print(f"  climatology grid {fclim.shape}, 80% source area ~ {area80:.1f} ha")
    print(f"  wrote {OUT}/{site}_ffp.npz")

if __name__=="__main__":
    main()
