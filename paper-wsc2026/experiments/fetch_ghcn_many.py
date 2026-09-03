"""Fetch MANY GHCN-Daily stations for the scale + heavy-tail experiments.

  data/conus_tmax_maxima.csv  year x station   annual JJA TMAX maxima (degC)
  data/conus_prcp_maxima.csv  year x station   annual max daily precip (mm)
  data/conus_meta.csv         station, lat, lon, elem

Uses NCEI's filtered data service (one element only, ~1 MB/station) via curl,
cached under data/cache/ so re-runs are instant and resumable.
"""
import csv
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

INV_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt"
DS = ("https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries"
      "&stations={sid}&dataTypes={elem}&startDate=1940-01-01&endDate=2025-12-31"
      "&units=metric&format=csv")
CONUS = dict(latmin=24.0, latmax=50.0, lonmin=-125.0, lonmax=-66.0)
N_TMAX, N_PRCP = 320, 120       # target station counts before the valid-years filter
MIN_YEARS = 50                  # keep a station only with this many valid annual maxima


def curl(url, out):
    subprocess.run(["curl", "-sS", "--retry", "3", "--max-time", "90", "-o", out, url],
                   check=True)


def inventory():
    inv = os.path.join(CACHE, "ghcnd-inventory.txt")
    if not os.path.exists(inv) or os.path.getsize(inv) < 1_000_000:
        print("downloading inventory ..."); curl(INV_URL, inv)
    return inv


def select(elem, n_target, seed, prefix=("USW", "USC")):
    rows = []
    for line in open(inventory()):
        sid, lat, lon = line[0:11], line[12:20], line[21:30]
        el, fy, ly = line[31:35].strip(), line[36:40], line[41:45]
        if el != elem or not sid.startswith(prefix):
            continue
        lat, lon, fy, ly = float(lat), float(lon), int(fy), int(ly)
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
    print(f"  selected {len(rows)} {elem} stations")
    return rows


def reduce_station(sid, elem, jja_only):
    out = os.path.join(CACHE, f"{sid}_{elem}.csv")
    if not os.path.exists(out) or os.path.getsize(out) < 50:
        try:
            curl(DS.format(sid=sid, elem=elem), out)
        except subprocess.CalledProcessError:
            return sid, {}
    vals, counts = {}, {}
    try:
        for row in csv.DictReader(open(out)):
            v = row.get(elem)
            d = row.get("DATE", "")
            if not v or len(d) < 7:
                continue
            if jja_only and d[5:7] not in ("06", "07", "08"):
                continue
            try:
                x = float(v)
            except ValueError:
                continue
            y = int(d[:4])
            vals[y] = max(vals.get(y, -9e9), x)
            counts[y] = counts.get(y, 0) + 1
    except (OSError, csv.Error):
        return sid, {}
    min_days = 80 if jja_only else 200
    maxima = {y: vals[y] for y in vals if counts[y] >= min_days and y <= 2025}
    return sid, maxima


def build(elem, jja_only, stations, fname):
    series, meta = {}, {}
    with ThreadPoolExecutor(max_workers=20) as ex:
        futs = {ex.submit(reduce_station, sid, elem, jja_only): (sid, lat, lon)
                for sid, lat, lon in stations}
        for i, fut in enumerate(futs):
            sid, lat, lon = futs[fut]
            _, maxima = fut.result()
            if len(maxima) >= MIN_YEARS:
                series[sid] = maxima
                meta[sid] = (lat, lon)
            if (i + 1) % 40 == 0:
                print(f"    {i+1}/{len(stations)} fetched, {len(series)} kept")
    years = sorted({y for s in series.values() for y in s})
    df = pd.DataFrame({sid: [series[sid].get(y, np.nan) for y in years]
                       for sid in series}, index=years)
    df.index.name = "year"
    df.to_csv(os.path.join(DATA, fname))
    print(f"  wrote data/{fname}: {df.shape[0]} years x {df.shape[1]} stations")
    return meta


def cached_stations(elem):
    """Reuse stations already fetched into the cache (no new network calls)."""
    import glob
    ids = {os.path.basename(f).split("_")[0]
           for f in glob.glob(os.path.join(CACHE, f"*_{elem}.csv"))}
    out = []
    for line in open(inventory()):
        sid = line[0:11]
        if line[31:35].strip() == elem and sid in ids:
            out.append((sid, float(line[12:20]), float(line[21:30])))
    print(f"  reusing {len(out)} cached {elem} stations")
    return out


def main():
    print("TMAX (scale demo, from cache):")
    tmeta = build("TMAX", True, cached_stations("TMAX"), "conus_tmax_maxima.csv")
    print("PRCP (heavy-tail demo):")
    pmeta = build("PRCP", False, select("PRCP", N_PRCP, seed=23), "conus_prcp_maxima.csv")
    rows = ([(s, la, lo, "TMAX") for s, (la, lo) in tmeta.items()]
            + [(s, la, lo, "PRCP") for s, (la, lo) in pmeta.items()])
    pd.DataFrame(rows, columns=["station", "lat", "lon", "elem"]).to_csv(
        os.path.join(DATA, "conus_meta.csv"), index=False)
    print(f"wrote data/conus_meta.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
