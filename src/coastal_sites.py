"""Classify AmeriFlux tower sites by proximity to the US coastline.

Reads the AmeriFlux site catalog and the Natural Earth 10m coastline, computes
each site's true point-to-segment distance to the nearest coastline (local
equirectangular projection, so distances are in metres not degrees), and writes
a tiered inventory of candidate coastal ET towers.

Run via Slurm, never on the login node.
"""

import json
import os
import csv
import math

import numpy as np

RAW = "/anvil/scratch/x-jwang120/coastal-et/data/raw"
OUT = "/anvil/scratch/x-jwang120/coastal-et/data/processed"

R_EARTH_KM = 6371.0088

# Ecosystems whose energy balance is genuinely coastal/tidal, vs. merely nearby.
TIDAL_IGBP = {"WET", "WAT"}

# US states/territories with a marine coastline (excludes Great Lakes-only states,
# which sit on fresh water and have no tidal regime).
COASTAL_STATES = {
    "AL", "AK", "CA", "CT", "DE", "FL", "GA", "HI", "LA", "ME", "MD", "MA",
    "MS", "NH", "NJ", "NY", "NC", "OR", "PA", "RI", "SC", "TX", "VA", "WA",
    "PR", "VI", "GU", "AS", "MP",
}


def load_coastline(path, bbox):
    """Return list of (lat, lon) vertex arrays for coastline segments in bbox."""
    with open(path) as f:
        gj = json.load(f)
    lon0, lat0, lon1, lat1 = bbox
    segs = []
    for feat in gj["features"]:
        geom = feat["geometry"]
        if geom["type"] == "LineString":
            parts = [geom["coordinates"]]
        elif geom["type"] == "MultiLineString":
            parts = geom["coordinates"]
        else:
            continue
        for coords in parts:
            arr = np.asarray(coords, dtype=float)
            if arr.ndim != 2 or arr.shape[0] < 2:
                continue
            lon, lat = arr[:, 0], arr[:, 1]
            # keep any segment that intersects the bbox at all
            if lon.max() < lon0 or lon.min() > lon1:
                continue
            if lat.max() < lat0 or lat.min() > lat1:
                continue
            segs.append(arr)
    return segs


def dist_to_coast_km(lat, lon, segs):
    """Minimum point-to-segment distance from (lat, lon) to any coastline segment."""
    coslat = math.cos(math.radians(lat))
    best = float("inf")
    for arr in segs:
        # local equirectangular projection in km, centred on the site
        x = (arr[:, 0] - lon) * coslat * math.pi / 180.0 * R_EARTH_KM
        y = (arr[:, 1] - lat) * math.pi / 180.0 * R_EARTH_KM

        # cheap reject: if the whole segment's bbox is farther than current best
        if min(np.abs(x).min(), np.abs(y).min()) > best:
            continue

        ax, ay = x[:-1], y[:-1]
        bx, by = x[1:], y[1:]
        dx, dy = bx - ax, by - ay
        seg_len2 = dx * dx + dy * dy
        seg_len2 = np.where(seg_len2 == 0, 1e-12, seg_len2)
        # projection parameter of the origin (the site) onto each segment
        t = np.clip(-(ax * dx + ay * dy) / seg_len2, 0.0, 1.0)
        px, py = ax + t * dx, ay + t * dy
        d = np.sqrt(px * px + py * py).min()
        if d < best:
            best = float(d)
    return best


def main():
    os.makedirs(OUT, exist_ok=True)

    with open(os.path.join(RAW, "ameriflux", "sites_ameriflux.json")) as f:
        sites = json.load(f)

    us = []
    for s in sites:
        if s.get("COUNTRY") != "USA":
            continue
        loc = s.get("GRP_LOCATION") or {}
        lat, lon = loc.get("LOCATION_LAT"), loc.get("LOCATION_LONG")
        if not lat or not lon:
            continue
        clim = s.get("GRP_CLIM_AVG") or {}
        us.append({
            "SITE_ID": s["SITE_ID"],
            "SITE_NAME": s.get("SITE_NAME", ""),
            "STATE": s.get("STATE", ""),
            "IGBP": s.get("IGBP", ""),
            "LAT": float(lat),
            "LON": float(lon),
            "ELEV": loc.get("LOCATION_ELEV", ""),
            "KOEPPEN": clim.get("CLIMATE_KOEPPEN", ""),
            "MAT": clim.get("MAT", ""),
            "MAP": clim.get("MAP", ""),
            "TOWER_BEGAN": s.get("TOWER_BEGAN", ""),
            "TOWER_END": s.get("TOWER_END", ""),
            "URL": s.get("URL_AMERIFLUX", ""),
        })

    print(f"AmeriFlux sites total     : {len(sites)}")
    print(f"US sites with coordinates : {len(us)}")

    lats = [s["LAT"] for s in us]
    lons = [s["LON"] for s in us]
    bbox = (min(lons) - 3, min(lats) - 3, max(lons) + 3, max(lats) + 3)
    segs = load_coastline(
        os.path.join(RAW, "coastline", "ne_10m_coastline.geojson"), bbox
    )
    print(f"Coastline segments in bbox: {len(segs)}")

    for s in us:
        s["DIST_COAST_KM"] = round(dist_to_coast_km(s["LAT"], s["LON"], segs), 3)
        s["COASTAL_STATE"] = s["STATE"] in COASTAL_STATES
        s["TIDAL_IGBP"] = s["IGBP"] in TIDAL_IGBP
        # active if the tower has no end year, or ended recently
        end = s["TOWER_END"]
        s["ACTIVE"] = (not end) or (end.isdigit() and int(end) >= 2020)

    us.sort(key=lambda s: s["DIST_COAST_KM"])

    fields = list(us[0].keys())
    with open(os.path.join(OUT, "us_sites_coastal_distance.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(us)

    # ---- tiered inventory -------------------------------------------------
    def sel(max_km, tidal_only=False, active_only=False):
        out = [s for s in us
               if s["DIST_COAST_KM"] <= max_km
               and s["COASTAL_STATE"]
               and (s["TIDAL_IGBP"] or not tidal_only)
               and (s["ACTIVE"] or not active_only)]
        return out

    print("\n" + "=" * 74)
    print("HOW MANY COASTAL FLUX TOWER SITES ARE THERE IN THE US?")
    print("=" * 74)
    print(f"{'buffer':>8} | {'all IGBP':>9} | {'tidal wet/water':>16} | {'tidal + active':>15}")
    print("-" * 74)
    for km in (1, 5, 10, 25, 50, 100):
        print(f"{km:>6} km | {len(sel(km)):>9} | {len(sel(km, tidal_only=True)):>16}"
              f" | {len(sel(km, tidal_only=True, active_only=True)):>15}")

    # IGBP breakdown inside the 10 km coastal zone
    print("\nIGBP breakdown within 10 km of coast (coastal states):")
    from collections import Counter
    c = Counter(s["IGBP"] or "(none)" for s in sel(10))
    for igbp, n in c.most_common():
        print(f"  {igbp:<8} {n:>3}")

    # the core training set: tidal wetland/water within 10 km
    core = sel(10, tidal_only=True)
    print(f"\nCORE COASTAL-WETLAND ET TOWER SET  (IGBP WET/WAT, <=10 km):  n = {len(core)}")
    print(f"{'SITE_ID':<10} {'ST':<3} {'IGBP':<5} {'km':>6}  {'years':<12} name")
    for s in core:
        yrs = f"{s['TOWER_BEGAN']}-{s['TOWER_END'] or 'now'}"
        print(f"{s['SITE_ID']:<10} {s['STATE']:<3} {s['IGBP']:<5} "
              f"{s['DIST_COAST_KM']:>6.1f}  {yrs:<12} {s['SITE_NAME'][:36]}")

    with open(os.path.join(OUT, "core_coastal_sites.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(core)

    print(f"\nwrote {OUT}/us_sites_coastal_distance.csv")
    print(f"wrote {OUT}/core_coastal_sites.csv")


if __name__ == "__main__":
    main()
