import pandas as pd 
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data/raw/power_27.67_85.429.csv", index_col = 0, parse_dates = True)

df["year"],df["month"] = df.index.year, df.index.month

df["hot_day"] = df["T2M_MAX"] > 40
df["frost_day"] = df["T2M_MIN"] <= 0
df["heavy_rain_day"] = df["PRECTOTCORR"] > 50

monthly = df.groupby(["year", "month"]).agg(
        rain_mm = ("PRECTOTCORR", "sum"),
        hot_days = ("hot_day", "sum"),
        frost_days=("frost_day", "sum"),
        heavy_rain_day = ("heavy_rain_day", sum),
        )

typical  = monthly.groupby("month")["rain_mm"].transform("median")

monthly["dry_month"] = (monthly["rain_mm"] < 0.7 * typical) & (typical >=30)

summary = monthly.groupby("month").agg(
        n_years=("rain_mm", "count"),
        typical_rain_nm = ("rain_mm", "median"),
        years_dry = ("dry_month", "sum"),
        median_hot_days = ("hot_days", "median"),
        max_hot_days = ("hot_days", "max"),
        years_with_heavy_rain = ("heavy_rain_day", lambda s : (s>0).sum()),
        year_with_frost = ("frost_days", lambda s: (s>0).sum()),
        )

print(summary.round(1))
summary.to_csv(ROOT / "data/processed/monthly_stress_summary.csv")
