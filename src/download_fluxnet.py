"""Download the AmeriFlux FLUXNET product (FULLSET) for every flux site we have.

Why this beats everything else we tried for ERA5:

  The FLUXNET package ships an AUX product -- AMF_<site>_FLUXNET_ERA5_HH/DD_*.csv
  -- which is ERA5 already downscaled and time-matched to the tower, hourly and
  daily, 1981-present. No CDS, no queue limits, no ocean-mask problem.

  It also ships LE_CORR (energy-balance-closed latent heat, done by ONEFlux) with
  published uncertainty (LE_CORR_25/75, LE_RANDUNC), which is a better reference
  ET than anything we can hand-roll from BASE.

The API needs data_product="FLUXNET" AND data_variant="FULLSET". Without the
variant it silently returns zero URLs -- a valid request that yields nothing.

FLUXNET is a CURATED subset: many BASE sites have no FLUXNET product at all.
Those sites keep their BASE-derived ET and gridMET forcing. We report which is
which rather than letting the mixture disappear into one column.

Run via Slurm, never on the login node.
"""

import os
import sys
import json
import time
import zipfile
import urllib.request
import io_utils  # idempotent skip/overwrite helper (same src/ dir)

API = "https://amfcdn.lbl.gov/api/v1/data_download"
UA = "coastal-et/0.1 (research; Purdue Anvil)"
OUT = "/anvil/scratch/x-jwang120/coastal-et/data/raw/ameriflux/fluxnet"
SITES_TXT = "/anvil/scratch/x-jwang120/coastal-et/config/flux_sites_all.txt"


def request_urls(sites, user, email):
    payload = {
        "user_id": user,
        "user_email": email,
        "data_product": "FLUXNET",
        "data_variant": "FULLSET",     # <-- without this the API returns nothing
        "data_policy": "CCBY4.0",
        "site_ids": sites,
        "intended_use": "Research - Land",
        "description": ("Upscaling eddy-covariance ET from coastal/wetland flux "
                        "towers using Landsat, Sentinel-2 and reanalysis."),
        "is_test": False,
    }
    req = urllib.request.Request(
        API, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": UA},
        method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())


def main():
    user = os.environ.get("AMF_USER", "")
    email = os.environ.get("AMF_EMAIL", "")
    if not user or not email or "CHANGEME" in email:
        sys.exit("set AMF_USER / AMF_EMAIL (see ~/.config/coastal-et/credentials.env)")

    sites = [l.strip() for l in open(SITES_TXT) if l.strip()]
    os.makedirs(OUT, exist_ok=True)
    print(f"requesting FLUXNET FULLSET for {len(sites)} sites")

    resp = request_urls(sites, user, email)
    urls = resp.get("data_urls") or []
    if isinstance(urls, str):
        urls = [urls]
    if not urls:
        print("no FLUXNET products returned. raw response:")
        print(json.dumps(resp, indent=2)[:1500])
        sys.exit(1)

    got = set()
    for i, item in enumerate(urls, 1):
        url = item["url"] if isinstance(item, dict) else item
        name = url.split("/")[-1].split("?")[0]
        site = name.split("_")[1]
        got.add(site)
        dest = os.path.join(OUT, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 10000 and not io_utils.force_requested():
            print(f"  [{i}/{len(urls)}] cached  {name}  (--force to redownload)")
        else:
            print(f"  [{i}/{len(urls)}] fetch   {name}", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=900) as r, open(dest, "wb") as f:
                while chunk := r.read(1 << 22):
                    f.write(chunk)
            time.sleep(1)
        try:
            with zipfile.ZipFile(dest) as z:
                z.extractall(os.path.join(OUT, site))
        except zipfile.BadZipFile:
            print(f"      BAD ZIP: {name}")

    missing = [s for s in sites if s not in got]
    print(f"\nFLUXNET available : {len(got)}/{len(sites)} sites")
    print(f"  {' '.join(sorted(got))}")
    print(f"\nNO FLUXNET product ({len(missing)}) -- these keep BASE ET + gridMET:")
    print(f"  {' '.join(missing)}")

    # confirm the ERA5 AUX actually arrived, since that is the whole point
    n_era = 0
    for s in sorted(got):
        era = [f for f in os.listdir(os.path.join(OUT, s))
               if "ERA5" in f and "_DD_" in f] if os.path.isdir(
                   os.path.join(OUT, s)) else []
        if era:
            n_era += 1
    print(f"\nERA5 AUX (daily) present for {n_era}/{len(got)} FLUXNET sites")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
