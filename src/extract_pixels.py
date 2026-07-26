"""Per-pixel satellite features on a grid around each tower, footprint-weighted.

For each tower and each Landsat overpass, keep EVERY 30 m pixel in a +/-300 m box
(not a window mean), with per-pixel LST/NDVI/MNDWI. Each pixel is tagged with its
Kljun-2015 footprint weight (interpolated from the climatology), so the tower's
footprint-integrated ET can be attributed to the pixels that actually produced it.

Output (long form, one row per pixel per overpass date):
  date, site, px_x, px_y (m from tower), fp_weight, LST_K, NDVI, MNDWI
"""
import os, sys, time
os.environ.setdefault("GDAL_HTTP_MAX_RETRY","6"); os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN","EMPTY_DIR")
import numpy as np, pandas as pd, dask, pyproj
import planetary_computer as pc, pystac_client, stackstac
from scipy.interpolate import RegularGridInterpolator
dask.config.set(scheduler="threads", num_workers=16)
R="/anvil/scratch/x-jwang120/coastal-et"
OUT=f"{R}/data/interim/pixels"; FP=f"{R}/data/interim/footprint"
FULL=f"{R}/data/processed/us_sites_coastal_distance.csv"
STAC="https://planetarycomputer.microsoft.com/api/stac/v1"
BOX_M=300.0; RES=30
ASSETS=["red","nir08","green","swir16","lwir11","qa_pixel"]

def coords(site):
    r=pd.read_csv(FULL).query("SITE_ID==@site").iloc[0]; return float(r.LAT),float(r.LON)

def fp_interp(site):
    d=np.load(f"{FP}/{site}_ffp.npz"); f=d["fclim"]; x=d["x"]; y=d["y"]
    xv=x[0] if x.ndim==2 else x; yv=y[:,0] if y.ndim==2 else y
    f=np.nan_to_num(f); f=f/f.sum() if f.sum()>0 else f
    return RegularGridInterpolator((yv,xv),f,bounds_error=False,fill_value=0.0)

def main():
    site=sys.argv[1]; lat,lon=coords(site); os.makedirs(OUT,exist_ok=True)
    epsg=32617   # UTM 17N (all Everglades sites)
    tx,ty=pyproj.Transformer.from_crs(4326,epsg,always_xy=True).transform(lon,lat)
    d=BOX_M/111320.0; dl=BOX_M/(111320.0*np.cos(np.radians(lat)))
    bbox=[lon-dl,lat-d,lon+dl,lat+d]
    fpi=fp_interp(site)
    cat=pystac_client.Client.open(STAC,modifier=pc.sign_inplace)
    rows=[]
    for yr in range(2018,2024):
        items=[i for i in cat.search(collections=["landsat-c2-l2"],bbox=bbox,
                 datetime=f"{yr}-01-01/{yr}-12-31",query={"eo:cloud_cover":{"lt":80}}).items()
               if set(ASSETS)<=set(i.assets)]
        if not items: continue
        for attempt in range(3):
            try:
                da=stackstac.stack(items,assets=ASSETS,bounds_latlon=bbox,epsg=epsg,
                                   resolution=RES,chunksize=256).compute(); break
            except Exception:
                if attempt==2: da=None
                else: time.sleep(5)
        if da is None: continue
        qa=da.sel(band="qa_pixel").values.astype("uint16")
        red=da.sel(band="red").values.astype("float32"); nir=da.sel(band="nir08").values.astype("float32")
        grn=da.sel(band="green").values.astype("float32"); swr=da.sel(band="swir16").values.astype("float32")
        lst=da.sel(band="lwir11").values.astype("float32")
        bad=(qa & 0b111110)>0
        for v in (red,nir,grn,swr,lst): v[bad]=np.nan
        # per-pixel indices with guards
        for arr in (red,nir,grn,swr): arr[arr<=0]=np.nan
        ndvi=(nir-red)/np.where((nir+red)>0.02,nir+red,np.nan); ndvi[np.abs(ndvi)>1]=np.nan
        mndwi=(grn-swr)/np.where((grn+swr)>0.02,grn+swr,np.nan); mndwi[np.abs(mndwi)>1]=np.nan
        savi=1.5*(nir-red)/np.where((nir+red+0.5)>0.05,nir+red+0.5,np.nan); savi[np.abs(savi)>1.5]=np.nan
        evi2=2.5*(nir-red)/np.where((nir+2.4*red+1)>0.05,nir+2.4*red+1,np.nan); evi2[np.abs(evi2)>1.5]=np.nan
        ndwi=(nir-swr)/np.where((nir+swr)>0.02,nir+swr,np.nan); ndwi[np.abs(ndwi)>1]=np.nan
        Xg,Yg=np.meshgrid(da.x.values,da.y.values)
        dx=(Xg-tx); dy=(Yg-ty)                      # metres E/N from tower
        w=fpi(np.stack([dy.ravel(),dx.ravel()],axis=1)).reshape(Xg.shape)
        times=pd.to_datetime(da.time.values,utc=True)
        for ti in range(da.sizes["time"]):
            L=lst[ti]
            if np.isfinite(L).mean()<0.3: continue    # too cloudy this scene
            m=np.isfinite(L)&(w>0)
            if m.sum()<3: continue
            rows.append(pd.DataFrame({
                "date":times[ti].normalize(),"px_x":dx[m],"px_y":dy[m],
                "fp_weight":w[m],"LST_K":L[m],"NDVI":ndvi[ti][m],"MNDWI":mndwi[ti][m],
                "SAVI":savi[ti][m],"EVI2":evi2[ti][m],"NDWI":ndwi[ti][m]}))
        print(f"  {site} {yr}: {len(items)} scenes",flush=True)
    if not rows: raise SystemExit(f"{site}: no pixels")
    df=pd.concat(rows,ignore_index=True); df["SITE_ID"]=site
    df.to_parquet(f"{OUT}/{site}_pixels.parquet")
    nd=df.date.nunique()
    print(f"{site}: {len(df):,} pixel-obs over {nd} overpass days, "
          f"fp_weight in [{df.fp_weight.min():.2e},{df.fp_weight.max():.2e}]")
    print(f"  wrote {OUT}/{site}_pixels.parquet")

if __name__=="__main__": main()
