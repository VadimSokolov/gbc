"""Fetch CONUS annual-maximum daily PRECIPITATION via the fast GHCN-Daily .dly files.

NCEI's filtered data service is currently very slow (~300 s/station); the static
.dly flat-file server (pub/data/ghcn/daily/all/) returns a whole station in ~1-2 s.
PRCP is stored in tenths of mm; we take the calendar-year maximum daily total,
keeping only quality-flag-blank values and years with >= 200 valid days.

  data/conus_prcp_maxima.csv   year x station   annual max daily precip (mm)
  data/conus_prcp_meta.csv     station, lat, lon, elem
"""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
CACHE = os.path.join(DATA, "cache")
os.makedirs(CACHE, exist_ok=True)
INV = os.path.join(CACHE, "ghcnd-inventory.txt")
DLY = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{sid}.dly"
CONUS = dict(latmin=24.0, latmax=50.0, lonmin=-125.0, lonmax=-66.0)
N_PRCP, MIN_YEARS, MIN_DAYS = 120, 50, 200


def curl_atomic(url, out, max_time=60):
    """Fetch to a temp file; only publish to cache on a clean, non-empty download."""
    tmp = out + ".tmp"
    r = subprocess.run(["curl", "-sS", "--max-time", str(max_time), "-o", tmp, url])
    if r.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 1000:
        os.replace(tmp, out)
        return True
    if os.path.exists(tmp):
        os.remove(tmp)
    return False


def select_prcp(n_target, seed=23, prefix=("USW", "USC")):
    rows = []
    for line in open(INV):
        sid, el = line[0:11], line[31:35].strip()
        if el != "PRCP" or not sid.startswith(prefix):
            continue
        lat, lon, fy, ly = float(line[12:20]), float(line[21:30]), int(line[36:40]), int(line[41:45])
        if not (CONUS["latmin"] <= lat <= CONUS["latmax"]
                and CONUS["lonmin"] <= lon <= CONUS["lonmax"]):
            continue
        if fy > 1965 or ly < 2020:           # long record, still reporting
            continue
        rows.append((sid, lat, lon))
    rng = np.random.default_rng(seed)
    if len(rows) > n_target:
        idx = sorted(rng.choice(len(rows), n_target, replace=False))
        rows = [rows[i] for i in idx]
    print(f"selected {len(rows)} CONUS PRCP stations")
    return rows


def prcp_maxima(sid):
    """Annual maximum daily precipitation (mm) from a station's .dly file."""
    out = os.path.join(CACHE, f"{sid}.dly")
    if not (os.path.exists(out) and os.path.getsize(out) > 1000):
        if not curl_atomic(DLY.format(sid=sid), out):
            return sid, {}
    vals, counts = {}, {}
    try:
        for line in open(out):
            if line[17:21] != "PRCP":
                continue
            year = int(line[11:15])
            if year > 2025:
                continue
            for d in range(31):
                off = 21 + d * 8
                raw = line[off:off + 5]
                if raw == "-9999" or line[off + 6:off + 7].strip():   # missing or QC-flagged
                    continue
                try:
                    v = int(raw) / 10.0                                # tenths of mm -> mm
                except ValueError:
                    continue
                if v < 0:
                    continue
                vals[year] = max(vals.get(year, 0.0), v)
                counts[year] = counts.get(year, 0) + 1
    except OSError:
        return sid, {}
    return sid, {y: vals[y] for y in vals if counts[y] >= MIN_DAYS}


def main():
    stations = select_prcp(N_PRCP)
    series, meta = {}, {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(prcp_maxima, sid): (sid, lat, lon) for sid, lat, lon in stations}
        for i, fut in enumerate(futs):
            sid, lat, lon = futs[fut]
            _, mx = fut.result()
            if len(mx) >= MIN_YEARS:
                series[sid] = mx
                meta[sid] = (lat, lon)
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(stations)} fetched, {len(series)} kept")
    years = sorted({y for s in series.values() for y in s})
    df = pd.DataFrame({sid: [series[sid].get(y, np.nan) for y in years]
                       for sid in series}, index=years)
    df.index.name = "year"
    df.to_csv(os.path.join(DATA, "conus_prcp_maxima.csv"))
    print(f"wrote data/conus_prcp_maxima.csv: {df.shape[0]} years x {df.shape[1]} stations")
    pd.DataFrame([(s, la, lo, "PRCP") for s, (la, lo) in meta.items()],
                 columns=["station", "lat", "lon", "elem"]).to_csv(
        os.path.join(DATA, "conus_prcp_meta.csv"), index=False)
    print(f"wrote data/conus_prcp_meta.csv ({len(meta)} stations)")


if __name__ == "__main__":
    main()
