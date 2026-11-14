"""Download daily NASA POWER data for one point and save it to data/raw/.

If the file already exists, only the parameters it is MISSING are downloaded and merged in,
so adding a name to PARAMS does not mean re-downloading everything.

Usage (run from the codebase/ folder):
    python src/fetch_power.py --lat 20.0 --lon 85.0
    python src/fetch_power.py --lat 20.0 --lon 85.0 --refresh     # re-download everything
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# ONE place to add parameters. Names must be valid for POWER's daily "AG" community
# (a single-point request allows at most 20 parameters).
PARAMS = [
    "T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "RH2M", "WS2M",
    "GWETROOT",      # root-zone soil wetness: 0 = dry ... 1 = saturated (about the top 1 m)
]
CHUNK_YEARS = 15     # fetch the long record in pieces, to stay clear of any request-size limit


def fetch_chunk(lat, lon, params_list, start, end):
    params = {
        "parameters": ",".join(params_list),   # commas only, no spaces
        "community": "AG",
        "latitude": lat,
        "longitude": lon,
        "start": start,
        "end": end,
        "format": "JSON",
    }
    r = requests.get(URL, params=params, timeout=180)
    if not r.ok:
        print(r.status_code, r.text[:500])     # POWER's error body names the bad parameter
    r.raise_for_status()
    data = r.json()["properties"]["parameter"]
    df = pd.DataFrame(data)                    # one column per variable, rows are dates
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    return df


def fetch_range(lat, lon, params_list, start, end):
    """start, end: pandas Timestamps. Downloads in CHUNK_YEARS pieces."""
    pieces, y0 = [], start.year
    while y0 <= end.year:
        y1 = min(y0 + CHUNK_YEARS - 1, end.year)
        s = start if y0 == start.year else pd.Timestamp(y0, 1, 1)
        e = end if y1 == end.year else pd.Timestamp(y1, 12, 31)
        print(f"fetching {params_list} {s.date()} -> {e.date()} ...")
        pieces.append(fetch_chunk(lat, lon, params_list, s.strftime("%Y%m%d"), e.strftime("%Y%m%d")))
        y0 = y1 + 1
    df = pd.concat(pieces).sort_index()
    df = df[~df.index.duplicated()]
    return df.replace(-999, np.nan).astype(float)      # POWER marks missing values as -999


def main():
    last_full_month_end = (pd.Timestamp.today().normalize() - pd.offsets.MonthEnd(1)).strftime("%Y%m%d")
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, default=20.0)
    ap.add_argument("--lon", type=float, default=85.0)
    ap.add_argument("--start-year", type=int, default=1981)
    ap.add_argument("--end", default=last_full_month_end, help="YYYYMMDD (default: end of last full month)")
    ap.add_argument("--refresh", action="store_true", help="ignore the saved file and download everything again")
    a = ap.parse_args()

    path = OUT_DIR / f"power_{a.lat}_{a.lon}.csv"
    if path.exists() and not a.refresh:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        missing = [p for p in PARAMS if p not in df.columns]
        if not missing:
            print(f"{path.name} already has every parameter in PARAMS; nothing to do "
                  f"({df.index.min().date()} -> {df.index.max().date()}). Use --refresh to re-download.")
            return
        print(f"{path.name} exists; adding only: {missing}")
        new = fetch_range(a.lat, a.lon, missing, df.index.min(), df.index.max())
        df = df.join(new, how="left")
    else:
        df = fetch_range(a.lat, a.lon, PARAMS, pd.Timestamp(a.start_year, 1, 1), pd.Timestamp(a.end))

    df.to_csv(path)
    print(f"saved {path}  shape={df.shape}  {df.index.min().date()} -> {df.index.max().date()}")
    print(df.isna().sum())


if __name__ == "__main__":
    main()
