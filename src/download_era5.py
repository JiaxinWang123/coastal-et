"""Download ERA5-Land hourly forcing for one regional box.

Why ERA5-Land and not the tower's own met: the pixels we upscale TO have no
tower. Training on tower-measured TA/VPD/Rn and then deploying on reanalysis
forcing is a distribution shift that quietly eats accuracy. So we train on
ERA5-Land from the start, and use tower met only to quantify reanalysis bias.

Why regional boxes: CDS costs a request by (variables x timesteps), not by area
-- measured against the live API, a 7x8 degree box and a 0.7 degree box at the
same time range cost the same. Grouping towers into 5 boxes turns ~17,000
requests into ~1,300.

Usage:  python download_era5.py REGION [START_YEAR] [END_YEAR]
Run via Slurm, never on the login node.
"""

import os
import sys
import time
import zipfile
import threading
import datetime as dt
import concurrent.futures as cf

sys.path.insert(0, "/anvil/scratch/x-jwang120/coastal-et/config")
from era5_regions import REGIONS, CHUNK_DAYS  # noqa: E402
import io_utils  # noqa: E402  (idempotent skip/overwrite helper, same src/ dir)

import cdsapi  # noqa: E402

RAW = "/anvil/scratch/x-jwang120/coastal-et/data/raw/era5"
DATASET = "reanalysis-era5-land"

# CDS caps how many requests a user may have QUEUED per dataset, and rejects the
# excess with a 400 "Number queued requests ... is temporarily limited". That is
# backpressure, not a bug -- the request is fine, we just asked too early. So we
# keep concurrency modest and treat that specific 400 as "wait and retry", with a
# long backoff, rather than burning a retry budget on it.
WORKERS = 2
RETRIES = 8
QUEUE_LIMIT_MSG = "temporarily limited"

INST = [
    "2m_temperature", "2m_dewpoint_temperature",
    "10m_u_component_of_wind", "10m_v_component_of_wind",
    "surface_pressure", "skin_temperature", "volumetric_soil_water_layer_1",
]
ACCUM = [
    "surface_net_solar_radiation", "surface_net_thermal_radiation",
    "surface_solar_radiation_downwards", "total_precipitation",
]
VARS = INST + ACCUM
HOURS = [f"{h:02d}:00" for h in range(24)]


def chunks(year):
    """Split a year into <=CHUNK_DAYS windows, never crossing a month boundary.

    CDS takes year/month/day lists, not a date range, so a chunk that spanned
    two months would request e.g. Feb 30. Staying inside a month avoids that.
    """
    out = []
    for m in range(1, 13):
        if m == 12:
            ndays = 31
        else:
            ndays = (dt.date(year, m + 1, 1) - dt.date(year, m, 1)).days
        d = 1
        while d <= ndays:
            hi = min(d + CHUNK_DAYS - 1, ndays)
            out.append((m, list(range(d, hi + 1))))
            d = hi + 1
    return out


def fetch_chunk(client, region, box, year, month, days):
    tag = f"{region}_{year}{month:02d}_{days[0]:02d}-{days[-1]:02d}"
    dest = os.path.join(RAW, f"{tag}.nc")
    if os.path.exists(dest) and os.path.getsize(dest) > 1000 and not io_utils.force_requested():
        return dest, True  # cached (--force to redownload)

    req = {
        "variable": VARS,
        "year": str(year),
        "month": f"{month:02d}",
        "day": [f"{d:02d}" for d in days],
        "time": HOURS,
        "area": box,                 # [N, W, S, E]
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    tmp = dest + ".tmp"
    client.retrieve(DATASET, req, tmp)

    if zipfile.is_zipfile(tmp):     # CDS sometimes zips anyway
        with zipfile.ZipFile(tmp) as z:
            nc = [n for n in z.namelist() if n.endswith(".nc")][0]
            with z.open(nc) as src, open(dest, "wb") as dst:
                dst.write(src.read())
        os.remove(tmp)
    else:
        os.rename(tmp, dest)
    return dest, False


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in REGIONS:
        raise SystemExit(f"usage: download_era5.py [{'|'.join(REGIONS)}] [y0] [y1]")
    region = sys.argv[1]
    cfg = REGIONS[region]
    box = cfg["box"]
    # default to the years this region's towers actually have flux data for
    dy0, dy1 = cfg["years"]
    y0 = int(sys.argv[2]) if len(sys.argv) > 2 else dy0
    y1 = int(sys.argv[3]) if len(sys.argv) > 3 else dy1
    os.makedirs(RAW, exist_ok=True)

    print(f"region {region}  box={box}  {y0}-{y1}")
    print(f"covers {len(cfg['sites'])} towers: {', '.join(cfg['sites'])}")

    work = [(y, m, d) for y in range(y0, y1 + 1) for m, d in chunks(y)]
    print(f"{len(work)} chunks, {WORKERS} concurrent")

    # A CDS request spends almost all its time QUEUED on their side, not
    # computing on ours. Serially that is ~10 min of pure waiting per chunk --
    # 40+ hours per region. Issuing them concurrently overlaps the waiting.
    done = cached = failed = 0
    bad = []
    lock = threading.Lock()

    def one(job):
        year, month, days = job
        client = cdsapi.Client(quiet=True)   # a Client per thread, not shared
        for attempt in range(1, RETRIES + 1):
            try:
                _, was_cached = fetch_chunk(client, region, box, year, month, days)
                return job, was_cached, None
            except Exception as e:
                msg = str(e)
                if attempt == RETRIES:
                    return job, False, msg.splitlines()[0][:60]
                # queue-limit rejections are pure backpressure: sleep it off,
                # long, and do not count it as a real failure
                if QUEUE_LIMIT_MSG in msg:
                    time.sleep(min(120 * attempt, 600))
                else:
                    time.sleep(20 * attempt)
        return job, False, "unreachable"

    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for job, was_cached, err in ex.map(one, work):
            year, month, days = job
            tag = f"{year}-{month:02d} d{days[0]:02d}-{days[-1]:02d}"
            with lock:
                if err:
                    failed += 1
                    bad.append(tag)
                    print(f"  {tag}  FAILED {err}", flush=True)
                elif was_cached:
                    cached += 1
                else:
                    done += 1
                    if done % 10 == 0:
                        print(f"  {tag}  ok  [{done + cached}/{len(work)}]", flush=True)

    print(f"\n{region}: {done} downloaded, {cached} cached, {failed} failed")
    if bad:
        # never let a gap pass silently -- a missing chunk is a hole in the forcing
        print(f"MISSING CHUNKS ({len(bad)}): {bad}")
        sys.exit(1)


if __name__ == "__main__":
    main()
