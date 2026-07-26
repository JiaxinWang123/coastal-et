"""Extract ECOSTRESS LST at each tower point, keeping acquisition hour.

ECOSTRESS L2T LSTE tiles (70 m, already Kelvin). Unlike Landsat's fixed ~10:30
overpass, ECOSTRESS samples LST across the day, so we KEEP the local solar hour
as a companion feature -- the same surface is several K hotter at 3pm than 8am,
and without the hour the mixed-time LST would just look like noise.

Window: 210 m (3x3 ECOSTRESS pixels), the analogue of the 90 m Landsat point.
Robust per-scene: skip tiles where the tower is outside the footprint or the
window is mostly nodata (cloud / gap).
"""
import glob, re, os
import numpy as np, pandas as pd, rasterio
from rasterio.warp import transform as rio_tf

D = "/anvil/projects/x-ees260113/team2/datasets/ECOSTRESS_LST"
OUT = "/anvil/scratch/x-jwang120/coastal-et/data/interim/ecostress"
COORDS = {"US-Elm": (25.5519, -80.7826), "US-Esm": (25.4379, -80.5946),
          "US-EvM": (25.3539, -80.3810), "US-Skr": (25.3629, -81.0776),
          "US-TaS": (25.1908, -80.6391)}
DIRMAP = {"US-EvM": "US-Evm", "US-TaS": "US-Tas"}   # folder spelling differs
WIN_M = 210
MIN_VALID = 0.5


def extract(site):
    lat, lon = COORDS[site]
    folder = DIRMAP.get(site, site)
    rows = []
    for f in sorted(glob.glob(f"{D}/{folder}/*.tif")):
        m = re.search(r"_LST_(\d{8})T(\d{6})", os.path.basename(f))
        if not m:
            continue
        ts = pd.to_datetime(m.group(1) + m.group(2), format="%Y%m%d%H%M%S", utc=True)
        with rasterio.open(f) as r:
            xs, ys = rio_tf("EPSG:4326", r.crs, [lon], [lat])
            row, col = r.index(xs[0], ys[0])
            if not (0 <= row < r.height and 0 <= col < r.width):
                continue                                   # tower outside tile
            rad = max(1, int(round(WIN_M / abs(r.res[0]) / 2)))
            r0, r1 = max(0, row - rad), min(r.height, row + rad + 1)
            c0, c1 = max(0, col - rad), min(r.width, col + rad + 1)
            win = r.read(1, window=((r0, r1), (c0, c1))).astype(float)
        v = win[(win > 200) & (win < 340)]                 # plausible LST (K)
        if v.size == 0 or v.size < MIN_VALID * win.size:
            continue
        # local solar hour: UTC + lon/15 (Everglades lon ~-80.7 -> UTC-5.4)
        solar_hr = (ts.hour + ts.minute / 60 + lon / 15.0) % 24
        rows.append(dict(time=ts, ECO_LST_K=float(v.mean()),
                         ECO_LST_sd=float(v.std()),
                         ECO_hour_utc=ts.hour + ts.minute / 60,
                         ECO_solar_hour=solar_hr))
    if not rows:
        return None
    df = pd.DataFrame(rows).set_index("time").sort_index()
    return df


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"{'site':<8}{'scenes':>8}{'LST med(K)':>12}{'solar-hr range':>18}")
    print("-" * 48)
    for s in COORDS:
        df = extract(s)
        if df is None:
            print(f"{s:<8}  none usable"); continue
        df.to_parquet(f"{OUT}/{s}_ecostress.parquet")
        print(f"{s:<8}{len(df):>8}{df.ECO_LST_K.median():>12.1f}"
              f"{f'{df.ECO_solar_hour.min():.1f}-{df.ECO_solar_hour.max():.1f}':>18}")
    print(f"\nwrote per-site ECOSTRESS to {OUT}")


if __name__ == "__main__":
    main()
