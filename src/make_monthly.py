"""Turn the daily POWER csv into a monthly table, plus a first look at trend and variability.

Usage (run from the codebase/ folder):
    python src/make_monthly.py
    python src/make_monthly.py --lat 20.0 --lon 85.0
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--lat", type=float, default=20.0)
ap.add_argument("--lon", type=float, default=85.0)
a = ap.parse_args()

src = ROOT / "data" / "raw" / f"power_{a.lat}_{a.lon}.csv"
df = pd.read_csv(src, index_col=0, parse_dates=True)
df["year"], df["month"] = df.index.year, df.index.month
df["hot_day_40"] = (df["T2M_MAX"] > 40).astype(int)
if "GWETROOT" not in df.columns:
    print("note: GWETROOT is not in the daily file, so soilw_root will be empty. "
          "Run  python src/fetch_power.py --lat ... --lon ...  to add it.\n")

g = df.groupby(["year", "month"])
monthly = pd.DataFrame({
    "temp_days": g["T2M"].count(),
    "rain_days": g["PRECTOTCORR"].count(),
    "temp_mean": g["T2M"].mean(),
    "tmax_mean": g["T2M_MAX"].mean(),
    "tmin_mean": g["T2M_MIN"].mean(),
    "rain_mm": g["PRECTOTCORR"].sum(min_count=1),
    "solar_mj": g["ALLSKY_SFC_SW_DWN"].mean(),
    "rh_pct": g["RH2M"].mean(),
    "wind_ms": g["WS2M"].mean(),
    "soilw_root": g["GWETROOT"].mean() if "GWETROOT" in df.columns else np.nan,   # root-zone wetness, 0-1
    "hot_days_40": g["hot_day_40"].sum(),
}).reset_index()

monthly["date"] = pd.to_datetime(dict(year=monthly["year"], month=monthly["month"], day=1))
dim = monthly["date"].dt.days_in_month

# keep only COMPLETE months, so a half-observed month never looks cooler or drier than it was
before = len(monthly)
monthly = monthly[(monthly["temp_days"] == dim) & (monthly["rain_days"] == dim)]
monthly = monthly.drop(columns=["temp_days", "rain_days"])
cols = ["date", "year", "month"] + [c for c in monthly.columns if c not in ("date", "year", "month")]
monthly = monthly[cols].reset_index(drop=True)

out = ROOT / "data" / "processed" / f"monthly_{a.lat}_{a.lon}.csv"
out.parent.mkdir(parents=True, exist_ok=True)
monthly.to_csv(out, index=False)

print(f"kept {len(monthly)} of {before} months  ({monthly['date'].min().date()} -> {monthly['date'].max().date()})")
print(f"saved {out}\n")

# ---- first look: how big is the warming trend, and how unpredictable is each month? ----
full = monthly.groupby("year").filter(lambda x: len(x) == 12)
annual = full.groupby("year")["temp_mean"].mean()
if len(annual) >= 5:
    slope = np.polyfit(annual.index, annual.values, 1)[0]
    print(f"annual mean temperature trend: {slope * 10:+.2f} degC per decade  ({len(annual)} complete years)\n")

clim = monthly.groupby("month").agg(
    n_years=("temp_mean", "count"),
    temp_mean=("temp_mean", "mean"),
    temp_sd=("temp_mean", "std"),
    rain_mean=("rain_mm", "mean"),
    rain_sd=("rain_mm", "std"),
)
# coefficient of variation: how large the year-to-year swings are relative to the typical value
clim["rain_cv"] = (clim["rain_sd"] / clim["rain_mean"]).where(clim["rain_mean"] > 5)
print(clim.round(2).to_string())
