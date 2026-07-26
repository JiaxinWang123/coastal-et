"""Download AmeriFlux BASE+BADM half-hourly flux data for the coastal site set.

Uses the AmeriFlux data-download API. This requires a registered AmeriFlux
account -- set AMF_USER and AMF_EMAIL in the environment. Registration is free
at https://ameriflux.lbl.gov/data/data-processing-pipelines/ and downloading
obliges you to the CC-BY-4.0 data policy (cite the site DOIs).

Run via Slurm, never on the login node.
"""

import csv
import json
import os
import sys
import time
import urllib.request
import zipfile

API = "https://amfcdn.lbl.gov/api/v1/data_download"
UA = "coastal-et/0.1 (research; Purdue Anvil)"
OUT = "/anvil/scratch/x-jwang120/coastal-et/data/raw/ameriflux/base"
SITES_CSV = "/anvil/scratch/x-jwang120/coastal-et/config/coastal_sites.csv"

INTENDED_USE = "Research - Land"
DESCRIPTION = (
    "Upscaling eddy-covariance evapotranspiration from coastal wetland flux "
    "towers to the coastal zone using satellite thermal/optical/SAR data and "
    "physics-informed machine learning."
)


def read_sites(strata):
    with open(SITES_CSV) as f:
        rows = [r for r in csv.DictReader(f)]
    if strata:
        rows = [r for r in rows if r["STRATUM"] in strata]
    return [r["SITE_ID"] for r in rows]


def request_urls(site_ids, user, email):
    payload = {
        "user_id": user,
        "user_email": email,
        "data_product": "BASE-BADM",
        "data_policy": "CCBY4.0",
        "site_ids": site_ids,
        "intended_use": INTENDED_USE,
        "description": DESCRIPTION,
        "is_test": False,
    }
    req = urllib.request.Request(
        API,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())


def main():
    user = os.environ.get("AMF_USER", "")
    email = os.environ.get("AMF_EMAIL", "")
    if not user or not email or "CHANGEME" in user or "CHANGEME" in email:
        sys.exit(
            "ERROR: AmeriFlux identity not set.\n"
            "  Register (free) at https://ameriflux.lbl.gov/data/download-data/\n"
            "  then fill in ~/.config/coastal-et/credentials.env:\n"
            "    export AMF_USER=your_username\n"
            "    export AMF_EMAIL=your@email.edu"
        )

    strata = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    sites = read_sites(strata)
    print(f"requesting BASE-BADM for {len(sites)} sites: {', '.join(sites)}")

    os.makedirs(OUT, exist_ok=True)
    resp = request_urls(sites, user, email)

    urls = resp.get("data_urls") or resp.get("data_url") or []
    if isinstance(urls, str):
        urls = [urls]
    if not urls:
        print("no download URLs returned; raw response follows:")
        print(json.dumps(resp, indent=2)[:4000])
        sys.exit(1)

    manifest = []
    for i, item in enumerate(urls, 1):
        url = item["url"] if isinstance(item, dict) else item
        name = url.split("/")[-1].split("?")[0]
        dest = os.path.join(OUT, name)
        if os.path.exists(dest):
            print(f"[{i}/{len(urls)}] cached  {name}")
        else:
            print(f"[{i}/{len(urls)}] fetch   {name}")
            # the file server 403s the default Python-urllib User-Agent
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            time.sleep(1)  # be polite to the AmeriFlux servers
        if dest.endswith(".zip"):
            with zipfile.ZipFile(dest) as z:
                z.extractall(OUT)
        manifest.append({"file": name, "url": url})

    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\ndone: {len(manifest)} archives in {OUT}")
    print("NOTE: cite each site's DOI (see the BADM files) in any publication.")


if __name__ == "__main__":
    main()
