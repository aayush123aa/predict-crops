"""Download monthly climate-index series (ENSO and the Indian Ocean Dipole) from NOAA PSL and
save them to data/raw/indices.csv, where forecast_model.py picks them up automatically.

Usage (run from the codebase/ folder):
    python src/fetch_indices.py
    python src/fetch_indices.py --refresh        # ignore saved files and download again

If an automatic download fails, the script tells you which page to open. Download the
"standard format" (.data) file there, save it as data/raw/indices/<name>.data, and run the
script again: saved files are used first.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
LOCAL_DIR = RAW_DIR / "indices"

INDICES = {
    "nino34": {     # El Nino / La Nina: Nino 3.4 sea-surface-temperature anomaly (deg C)
        "urls": [
            "https://psl.noaa.gov/data/correlation/nina34.anom.data",                # ERSSTv5, linked from PSL's index page
            "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/nino34.long.anom.data",  # HadISST, older location (unverified)
        ],
        "page": "https://psl.noaa.gov/data/timeseries/month/Nino34_CPC/",
    },
    "dmi": {        # Indian Ocean Dipole: Dipole Mode Index (deg C)
        "urls": [
            "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data",      # older location (unverified)
        ],
        "page": "https://psl.noaa.gov/data/timeseries/month/DS/DMI/",
    },
}


def parse_psl(text):
    """Parse NOAA PSL 'standard format':
         first line : first_year last_year
         then       : year v1 ... v12   (one line per year)
         then       : a line holding just the missing-value marker
         then       : free-text info lines (ignored)
    Returns a monthly pandas Series (month-start index) with missing values as NaN."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    y1, y2 = (int(x) for x in lines[0].split()[:2])
    rows, missing = {}, None
    for ln in lines[1:]:
        tok = ln.split()
        if len(tok) == 13 and tok[0].isdigit() and y1 <= int(tok[0]) <= y2:
            rows[int(tok[0])] = [float(t) for t in tok[1:]]
        elif len(tok) == 1:
            try:
                missing = float(tok[0])
            except ValueError:
                pass
            break                                   # everything after this line is free text
    if not rows:
        raise ValueError("no data rows found; is this a PSL standard-format file?")
    dates, vals = [], []
    for yr in sorted(rows):
        for mo, v in enumerate(rows[yr], start=1):
            dates.append(pd.Timestamp(yr, mo, 1))
            vals.append(v)
    s = pd.Series(vals, index=pd.DatetimeIndex(dates), dtype=float)
    if missing is not None:
        s[np.isclose(s, missing)] = np.nan
    s[s < -90] = np.nan                             # safety net: no climate index is anywhere near -90
    return s


def get_series(name, spec, refresh):
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    local = LOCAL_DIR / f"{name}.data"
    if local.exists() and not refresh:
        print(f"{name}: using saved file {local}")
        return parse_psl(local.read_text())
    for url in spec["urls"]:
        try:
            r = requests.get(url, timeout=60)
            if not r.ok:
                print(f"{name}: HTTP {r.status_code} from {url}")
                continue
            s = parse_psl(r.text)
        except Exception as e:
            print(f"{name}: could not use {url} ({e})")
            continue
        local.write_text(r.text)
        print(f"{name}: downloaded {url}")
        return s
    print(f"{name}: automatic download failed.\n"
          f"   Open {spec['page']} in a browser, download the standard-format (.data) file,\n"
          f"   save it as {local}\n   and run this script again.")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()

    cols = {}
    for name, spec in INDICES.items():
        s = get_series(name, spec, a.refresh)
        if s is not None:
            cols[name] = s
    if not cols:
        print("\nno index data available; forecast_model.py will run without climate indices.")
        return
    df = pd.DataFrame(cols)
    df = df[df.index >= "1950-01-01"]
    df.index.name = "date"
    out = RAW_DIR / "indices.csv"
    df.to_csv(out)
    print(f"\nsaved {out}")
    for name in df.columns:
        v = df[name].dropna()
        print(f"  {name}: {v.index.min():%Y-%m} -> {v.index.max():%Y-%m}   latest value {v.iloc[-1]:+.2f}")


if __name__ == "__main__":
    main()
