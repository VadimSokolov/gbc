"""Fetch real PNW station extremes (GHCN-Daily) and the HadCRUT5 climate covariate.

Outputs:
  data/pnw_jja_maxima.csv   year, SEA, PDX, GEG, EUG, PDT  (JJA TMAX maxima, degC)
  data/hadcrut5_annual.csv  year, anomaly                  (global mean, degC vs 1961-90)

TMAX in the GHCN-Daily access CSVs is in tenths of a degree C.
A station-year counts only if it has >= MIN_JJA_DAYS valid June/July/August TMAX days.
"""

import csv
import io
import os
import urllib.request

import numpy as np
import pandas as pd

GHCND = "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/{}.csv"
HADCRUT5 = ("https://www.metoffice.gov.uk/hadobs/hadcrut5/data/HadCRUT.5.0.2.0/"
            "analysis/diagnostics/HadCRUT.5.0.2.0.analysis.summary_series.global.annual.csv")

STATIONS = {  # code -> (GHCND id, expected name fragment)
    "SEA": ("USW00024233", "SEATTLE TACOMA"),
    "PDX": ("USW00024229", "PORTLAND"),
    "GEG": ("USW00024157", "SPOKANE"),
    "EUG": ("USW00024221", "EUGENE"),
    "PDT": ("USW00024155", "PENDLETON"),
}
MIN_JJA_DAYS = 80
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")


def fetch_station(ghcnd_id, name_frag):
    text = urllib.request.urlopen(GHCND.format(ghcnd_id), timeout=120).read().decode()
    reader = csv.DictReader(io.StringIO(text))
    vals, counts, name = {}, {}, None
    for row in reader:
        name = row.get("NAME", name)
        date = row["DATE"]
        if date[5:7] not in ("06", "07", "08") or not row.get("TMAX"):
            continue
        try:
            v = float(row["TMAX"]) / 10.0
        except ValueError:
            continue
        y = int(date[:4])
        vals[y] = max(vals.get(y, -99.0), v)
        counts[y] = counts.get(y, 0) + 1
    assert name and name_frag in name, f"{ghcnd_id}: name {name!r} lacks {name_frag!r}"
    maxima = {y: vals[y] for y in vals if counts[y] >= MIN_JJA_DAYS}
    return name, maxima


def fetch_hadcrut5():
    text = urllib.request.urlopen(HADCRUT5, timeout=120).read().decode()
    reader = csv.DictReader(io.StringIO(text))
    out = {}
    for row in reader:
        # columns: Time, Anomaly (deg C), Lower confidence..., Upper...
        y = int(str(row["Time"])[:4])
        anom_key = [k for k in row if "Anomaly" in k][0]
        out[y] = float(row[anom_key])
    return out


def main():
    os.makedirs(DATA, exist_ok=True)
    series = {}
    for code, (gid, frag) in STATIONS.items():
        name, maxima = fetch_station(gid, frag)
        series[code] = maxima
        yrs = sorted(maxima)
        print(f"{code} {gid} {name:32s} {yrs[0]}-{yrs[-1]}  n={len(yrs)}  "
              f"max={max(maxima.values()):.1f}C")

    # Sanity check against the known 2021 heat dome.
    assert abs(series["SEA"][2021] - 42.2) < 0.05, series["SEA"][2021]
    print(f"\nVALIDATION  SEA 2021 = {series['SEA'][2021]:.1f}C (2021 heat dome, expect 42.2) OK")

    all_years = sorted(set().union(*[set(s) for s in series.values()]))
    all_years = [y for y in all_years if y <= 2025]  # drop partial 2026
    df = pd.DataFrame({code: [series[code].get(y, np.nan) for y in all_years]
                       for code in STATIONS}, index=all_years)
    df.index.name = "year"
    df.to_csv(os.path.join(DATA, "pnw_jja_maxima.csv"))
    print(f"\nwrote data/pnw_jja_maxima.csv  ({len(df)} years x {df.shape[1]} stations)")
    print(df.tail(4).round(1).to_string())

    had = fetch_hadcrut5()
    hdf = pd.DataFrame({"anomaly": [had[y] for y in sorted(had)]},
                       index=sorted(had))
    hdf.index.name = "year"
    hdf.to_csv(os.path.join(DATA, "hadcrut5_annual.csv"))
    print(f"\nwrote data/hadcrut5_annual.csv  ({hdf.index.min()}-{hdf.index.max()}, "
          f"2021 anomaly={had[2021]:.3f}C)")


if __name__ == "__main__":
    main()
