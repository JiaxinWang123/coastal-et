"""Download NOAA CO-OPS water level (tide) records for each coastal flux tower.

Tidal stage is the covariate that separates a coastal ET model from a generic
one: satellite overpasses are at a fixed solar time but the marsh surface may be
inundated or exposed depending on the tide, so the same LST/NDVI can correspond
to very different energy partitioning. We pair each tower with its nearest CO-OPS
gauge and derive inundation-relevant predictors.

CO-OPS API is public, no key required.
Run via Slurm, never on the login node.
"""

import json
import os
import math
import urllib.request
import io_utils  # idempotent skip/overwrite helper (same src/ dir)

import pandas as pd

OUT = "/anvil/scratch/x-jwang120/coastal-et/data/raw/tides"
SITES = "/anvil/scratch/x-jwang120/coastal-et/config/coastal_sites.csv"
FULL = "/anvil/scratch/x-jwang120/coastal-et/data/processed/us_sites_coastal_distance.csv"

MDAPI = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"
# note the /prod/ path: the bare /api/datagetter endpoint 403s
DATA = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

R = 6371.0088


def haversine(lat1, lon1, lat2, lon2):
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p)
         * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def gauges():
    url = f"{MDAPI}?type=waterlevels"
    with urllib.request.urlopen(url, timeout=120) as r:
        d = json.loads(r.read().decode())
    return [
        {"id": s["id"], "name": s["name"], "lat": s["lat"], "lon": s["lng"]}
        for s in d["stations"]
        if s.get("lat") and s.get("lng")
    ]


def nearest(lat, lon, gs):
    best = min(gs, key=lambda g: haversine(lat, lon, g["lat"], g["lon"]))
    return best, haversine(lat, lon, best["lat"], best["lon"])


def water_level(station, begin, end):
    """Fetch verified water level, MLLW datum, in metres, one year per request.

    Uses hourly_height, not water_level: CO-OPS caps the 6-minute water_level
    product at 31 days per request, while hourly_height allows a full year.
    The tidal signal is smooth (semidiurnal, ~12.4 h), so hourly sampling
    resolves it fine and we interpolate onto the 30-min flux grid downstream.
    """
    frames = []
    for yr in range(int(begin[:4]), int(end[:4]) + 1):
        params = (
            f"?product=hourly_height&application=coastal-et&begin_date={yr}0101"
            f"&end_date={yr}1231&datum=MLLW&station={station}"
            f"&time_zone=gmt&units=metric&format=json"
        )
        try:
            with urllib.request.urlopen(DATA + params, timeout=180) as r:
                d = json.loads(r.read().decode())
        except Exception as e:
            print(f"    {yr}: request failed ({e})")
            continue
        if "data" not in d:
            print(f"    {yr}: {d.get('error', {}).get('message', 'no data')}")
            continue
        df = pd.DataFrame(d["data"])
        df["t"] = pd.to_datetime(df["t"], utc=True)
        df["v"] = pd.to_numeric(df["v"], errors="coerce")
        frames.append(df[["t", "v"]].rename(columns={"v": "water_level_m"}))
        print(f"    {yr}: {len(df)} obs")
    return pd.concat(frames).set_index("t").sort_index() if frames else None


def derive(wl):
    """Turn a raw water-level series into the predictors the ET model needs."""
    # hourly -> 30-min flux grid. Interpolate, but only across short gaps: a
    # 3-slot limit spans 90 min, so real outages stay NaN instead of being
    # silently invented.
    h = wl.resample("30min").mean()
    h["water_level_m"] = h["water_level_m"].interpolate(limit=3, limit_area="inside")
    h["wl_rate_m_per_hr"] = h["water_level_m"].diff() * 2  # flooding vs ebbing
    # rolling stats capture inundation *history*, which drives soil moisture
    # and salinity memory that instantaneous tide stage misses
    h["wl_mean_24h"] = h["water_level_m"].rolling(48, min_periods=1).mean()
    h["wl_max_24h"] = h["water_level_m"].rolling(48, min_periods=1).max()
    h["wl_mean_7d"] = h["water_level_m"].rolling(336, min_periods=1).mean()
    hi = h["water_level_m"].quantile(0.90)
    h["inundation_frac_24h"] = (
        (h["water_level_m"] > hi).rolling(48, min_periods=1).mean()
    )
    return h


def main():
    os.makedirs(OUT, exist_ok=True)
    sites = pd.read_csv(SITES)
    coords = pd.read_csv(FULL)[["SITE_ID", "LAT", "LON"]]
    sites = sites.merge(coords, on="SITE_ID", how="left")
    sites = sites[sites.USE != "exclude"]

    gs = gauges()
    print(f"{len(gs)} NOAA CO-OPS water-level gauges\n")

    pairing = []
    for _, s in sites.iterrows():
        g, d = nearest(s.LAT, s.LON, gs)
        print(f"{s.SITE_ID:<8} -> gauge {g['id']} {g['name'][:32]:<34} {d:6.1f} km")
        pairing.append({
            "SITE_ID": s.SITE_ID, "STRATUM": s.STRATUM,
            "gauge_id": g["id"], "gauge_name": g["name"],
            "gauge_dist_km": round(d, 2),
        })

        begin = f"{int(s.TOWER_BEGAN)}-01-01" if pd.notna(s.TOWER_BEGAN) else "2015-01-01"
        end = f"{int(s.TOWER_END)}-12-31" if pd.notna(s.TOWER_END) else "2025-12-31"
        dest = os.path.join(OUT, f"{s.SITE_ID}_tide.parquet")
        if os.path.exists(dest) and not io_utils.force_requested():
            print("    cached (--force to redownload)")
            continue
        wl = water_level(g["id"], begin, end)
        if wl is None or wl.empty:
            print("    no water level data")
            continue
        derive(wl).to_parquet(dest)

    pd.DataFrame(pairing).to_csv(os.path.join(OUT, "site_gauge_pairing.csv"), index=False)
    print(f"\nwrote {OUT}/site_gauge_pairing.csv")
    print("REVIEW gauge_dist_km: a gauge >30 km away in a different sub-estuary")
    print("may have a materially different tidal phase and amplitude.")


if __name__ == "__main__":
    main()
