"""Forecast monthly temperature, rainfall and root-zone soil moisture with LightGBM, judged against simple baselines.

Usage (run from the codebase/ folder, after make_monthly.py):
    python src/forecast_model.py --lat 20.0 --lon 85.0
    python src/forecast_model.py --lat 20.0 --lon 85.0 --horizon 12 --test-years 15

What it does
  0. Chooses the baseline "usual value" for each calendar month: the last 10/15/20/30 years or
     all years, with or without a linear trend, by walk-forward test of the baseline alone.
  1. Splits every variable into: usual value (baseline) + anomaly (what is left).
  2. Trains LightGBM to predict the ANOMALY 1..H months ahead.
  3. Walk-forward test: for each test year Y, train only on data before Y, then predict year Y.
  4. Keeps LightGBM only at leads where it beats BOTH baselines (the chosen one and the all-years one).
  5. Writes a forecast whose 80% ranges come from the walk-forward errors, and warns if the
     point forecast falls outside its own range (a sign the baseline is biased).
"""
import argparse
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
INDEX_FILE = ROOT / "data" / "raw" / "indices.csv"   # written by fetch_indices.py

TARGET_VARS = ["temp_mean", "rain_mm", "soilw_root"]            # what we forecast (EcoCrop needs the first two)
FEATURE_VARS = ["temp_mean", "rain_mm", "solar_mj", "rh_pct", "soilw_root"]   # what the model may look at
ALL_VARS = sorted(set(TARGET_VARS) | set(FEATURE_VARS))
WINDOWS = [10, 15, 20, 30, 0]   # candidate years of history for the "usual value"; 0 = all years
CANDIDATES = [(w, tr) for w in WINDOWS for tr in (False, True)]   # (window years, add a linear trend?)
BOUNDS = {"rain_mm": (0.0, None), "soilw_root": (0.0, 1.0)}   # physical limits for forecasts and ranges
BASELINE_TOL = 0.05   # use the SIMPLEST baseline whose error is within 5% of the best one (trends can win by luck)
MIN_SKILL = 0.05   # LightGBM must cut squared error by >5% vs the better baseline (few test years: be strict)

LGB_PARAMS = dict(
    n_estimators=300, learning_rate=0.03,
    num_leaves=8, min_child_samples=30,          # small trees: we have little data
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_lambda=1.0, random_state=0, verbose=-1,
)


# ---------------------------------------------------------------- data
def load_monthly(lat, lon, exclude=()):
    """Reads the monthly table. Variables that are missing or empty are dropped (with a note).
    `exclude` removes FEATURES only: an excluded variable can still be forecast as a target."""
    global FEATURE_VARS, TARGET_VARS, ALL_VARS
    path = ROOT / "data" / "processed" / f"monthly_{lat}_{lon}.csv"
    m = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    m = m.reindex(pd.date_range(m.index.min(), m.index.max(), freq="MS"))   # gaps become explicit NaN
    wanted = sorted(set(FEATURE_VARS) | set(TARGET_VARS))
    empty = [v for v in wanted if v not in m.columns or m[v].isna().all()]
    if empty:
        print(f"note: no data for {empty}; continuing without them "
              "(run fetch_power.py then make_monthly.py to add them)")
    TARGET_VARS = [v for v in TARGET_VARS if v not in empty]
    FEATURE_VARS = [v for v in FEATURE_VARS if v not in empty and v not in exclude]
    ALL_VARS = sorted(set(TARGET_VARS) | set(FEATURE_VARS))
    return m[ALL_VARS]


def load_indices(monthly_index, exclude=()):
    """Climate-index features (ENSO, Indian Ocean Dipole) aligned to the monthly table.
    Each index is shifted by its own publication delay (at least 1 month), so that training
    and a real forecast made today see data with the same lag."""
    if not INDEX_FILE.exists():
        print("note: data/raw/indices.csv not found; continuing without climate indices "
              "(run fetch_indices.py to add them)")
        return None
    ix = pd.read_csv(INDEX_FILE, parse_dates=["date"]).set_index("date")
    last = monthly_index.max()
    out = {}
    for name in ix.columns:
        if name in exclude or ix[name].notna().sum() < 120:
            continue
        s = ix[name].reindex(pd.date_range(ix.index.min(), max(ix.index.max(), last), freq="MS"))
        lv = s.last_valid_index()
        lag = max(1, (last.year - lv.year) * 12 + (last.month - lv.month))
        print(f"  index {name}: latest value {lv:%Y-%m} -> used {lag} month(s) behind the forecast origin")
        out[f"{name}_L{lag}"] = s.shift(lag).reindex(monthly_index)
        out[f"{name}_m3_L{lag}"] = s.rolling(3, min_periods=2).mean().shift(lag).reindex(monthly_index)
    return pd.DataFrame(out, index=monthly_index) if out else None


# ---------------------------------------------------------------- baseline
def tdec(idx):
    """Time in years, centred on 2000, so an optional trend fit is well behaved."""
    return np.asarray(idx.year + (idx.month - 0.5) / 12 - 2000, dtype=float)


def label(c):
    w, tr = c
    return ("all years" if w == 0 else f"last {w}y") + (" +trend" if tr else "")


def fit_baseline(m, train_end, spec):
    """Per variable: month-of-year means over the last w years before train_end (0 = all years),
    optionally plus a linear trend, where spec[v] = (w, use_trend). Variables not in spec use
    all years and no trend. Uses ONLY months before train_end, so nothing leaks."""
    base = {}
    for v in ALL_VARS:
        w, use_trend = spec.get(v, (0, False))
        tr = m[m.index < train_end]
        if w:
            tr = tr[tr.index >= train_end - pd.DateOffset(years=w)]
        clim = tr.groupby(tr.index.month)[v].mean()
        slope = intercept = 0.0
        if use_trend:
            a = tr[v].to_numpy() - clim.reindex(tr.index.month).to_numpy()
            ok = ~np.isnan(a)
            slope, intercept = np.polyfit(tdec(tr.index[ok]), a[ok], 1)
        base[v] = (clim, slope, intercept)
    return base


def base_value(base, v, idx):
    clim, slope, intercept = base[v]
    out = clim.reindex(idx.month).to_numpy(dtype=float)
    if slope != 0.0:
        out = out + intercept + slope * tdec(idx)
    return out


def anomalies(m, base):
    return pd.DataFrame({v: m[v].to_numpy() - base_value(base, v, m.index) for v in ALL_VARS},
                        index=m.index)


def pick_baselines(m, test_years):
    """Walk-forward test of the baseline alone (no machine learning): which (window, trend)
    choice gives the smallest squared error, separately for each target variable?"""
    rows = []
    for c in CANDIDATES:
        se = {v: 0.0 for v in TARGET_VARS}
        n = {v: 0 for v in TARGET_VARS}
        for Y in test_years:
            base = fit_baseline(m, pd.Timestamp(Y, 1, 1), {v: c for v in TARGET_VARS})
            te = m[m.index.year == Y]
            for v in TARGET_VARS:
                err = te[v].to_numpy() - base_value(base, v, te.index)
                err = err[~np.isnan(err)]
                se[v] += float((err ** 2).sum())
                n[v] += len(err)
        rows.append(pd.DataFrame({"var": TARGET_VARS, "label": label(c),
                                  "mse": [se[v] / n[v] for v in TARGET_VARS]}))
    t = pd.concat(rows, ignore_index=True)
    # Prefer simplicity: no trend before trend, and longer windows before shorter ones.
    ranked = [0] + sorted([w for w in WINDOWS if w], reverse=True)
    order = sorted(CANDIDATES, key=lambda c: (c[1], ranked.index(c[0])))
    best = {}
    for v in TARGET_VARS:
        mv = t[t["var"] == v].set_index("label")["mse"]
        limit = mv.min() * (1 + BASELINE_TOL)
        best[v] = next(c for c in order if mv[label(c)] <= limit)
    return best, t


# ---------------------------------------------------------------- features
def origin_features(an, extra=None):
    """What the model knows at the end of an 'origin' month: recent anomalies,
    plus (optionally) climate-index columns that are already lagged by their publication delay."""
    f = pd.DataFrame(index=an.index)
    for v in FEATURE_VARS:
        s = an[v]
        f[f"{v}_a0"] = s                                   # this month
        f[f"{v}_a1"] = s.shift(1)                          # last month
        f[f"{v}_a2"] = s.shift(2)                          # two months ago
        f[f"{v}_m3"] = s.rolling(3, min_periods=2).mean()  # 3-month average
    f["origin_month"] = f.index.month
    if extra is not None:
        f = f.join(extra)
    return f


def stack(an, feats, target_var, horizon):
    """One row per (origin month, lead). Target = anomaly of the month `lead` months later."""
    parts = []
    for h in range(1, horizon + 1):
        d = feats.copy()
        d["lead"] = h
        tgt = d.index + pd.DateOffset(months=h)
        d["tgt_date"] = tgt
        d["tgt_month"] = tgt.month
        d["y_anom"] = an[target_var].reindex(tgt).to_numpy()   # NaN if that month is not observed yet
        parts.append(d)
    return pd.concat(parts)


def feat_cols(S):
    return [c for c in S.columns if c not in ("tgt_date", "y_anom")]


def mse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, default=20.0)
    ap.add_argument("--lon", type=float, default=85.0)
    ap.add_argument("--horizon", type=int, default=12, help="months ahead to forecast")
    ap.add_argument("--test-years", type=int, default=15, help="how many recent years to test on")
    ap.add_argument("--exclude", nargs="*", default=[], help="features to leave out, e.g. --exclude soilw_root nino34 dmi")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. --tag nosoil")
    args = ap.parse_args()
    sfx = f"_{args.tag}" if args.tag else ""

    m = load_monthly(args.lat, args.lon, args.exclude)
    extra = load_indices(m.index, args.exclude)
    print(f"features used: {FEATURE_VARS}" + (f" + climate indices {list(extra.columns)}" if extra is not None else ""))
    last = m.index.max()
    years = sorted(m.index.year.unique())
    first_test = max(years[0] + 15, years[-1] - args.test_years + 1)   # keep >= 15 years for training
    test_years = list(range(first_test, years[-1] + 1))
    print(f"data: {m.index.min().date()} -> {last.date()}  ({len(m)} months)")
    print(f"walk-forward test years: {test_years[0]}-{test_years[-1]}   horizon: {args.horizon} months\n")

    # ---------- 0) choose the baseline (window of recent years, with or without a trend) per variable
    spec, wtab = pick_baselines(m, test_years)
    piv = wtab.pivot(index="var", columns="label", values="mse")[[label(c) for c in CANDIDATES]]
    rel = piv.div(piv["all years"], axis=0)
    print("BASELINE CHOICE: squared error of the 'usual value' alone, relative to all years (1.00)")
    print(f"  below 1.00 = better than the full record; the simplest choice within {BASELINE_TOL:.0%} of the best is used\n")
    print(rel.round(3).T.to_string())
    print("\n  chosen:", {v: label(c) for v, c in spec.items()})
    print()

    # ---------- 1) walk-forward evaluation
    rows = []
    for Y in test_years:
        train_end = pd.Timestamp(Y, 1, 1)
        base = fit_baseline(m, train_end, spec)        # chosen baseline
        base_all = fit_baseline(m, train_end, {})      # all years, for comparison
        an = anomalies(m, base)
        feats = origin_features(an, extra)
        for v in TARGET_VARS:
            S = stack(an, feats, v, args.horizon)
            cols = feat_cols(S)
            train = S[(S["tgt_date"] < train_end) & S["y_anom"].notna()]
            test = S[(S["tgt_date"].dt.year == Y) & S["y_anom"].notna()]
            if len(test) == 0:
                continue
            model = lgb.LGBMRegressor(**LGB_PARAMS).fit(train[cols], train["y_anom"])
            td = pd.DatetimeIndex(test["tgt_date"])
            b1 = base_value(base, v, td)
            rows.append(pd.DataFrame({
                "var": v, "lead": test["lead"].to_numpy(), "tgt_date": td,
                "y_true": m[v].reindex(td).to_numpy(),
                "b0": base_value(base_all, v, td),               # baseline 0: all-years usual value
                "b1": b1,                                        # baseline 1: chosen-window usual value
                "lgbm": b1 + model.predict(test[cols]),          # chosen baseline + LightGBM anomaly
            }))
    res = pd.concat(rows, ignore_index=True)

    # ---------- 2) skill table: positive = better than the baseline it is compared with
    out_rows = []
    for (v, lead), g in res.groupby(["var", "lead"]):
        e0, e1, e2 = mse(g.y_true, g.b0), mse(g.y_true, g.b1), mse(g.y_true, g.lgbm)
        out_rows.append(dict(var=v, lead=lead, n=len(g),
                             base_vs_all=1 - e1 / e0,
                             lgbm_vs_all=1 - e2 / e0,
                             lgbm_vs_base=1 - e2 / e1))
    skill = pd.DataFrame(out_rows)
    # LightGBM must beat the BETTER of the two baselines, i.e. both comparisons must clear the bar
    skill["use_lgbm"] = skill[["lgbm_vs_all", "lgbm_vs_base"]].min(axis=1) > MIN_SKILL

    proc = ROOT / "data" / "processed"
    skill.to_csv(proc / f"skill_{args.lat}_{args.lon}{sfx}.csv", index=False)
    print("SKILL (walk-forward; >0 means better than that baseline)")
    print("  base_vs_all : chosen baseline vs all-years baseline")
    print("  lgbm_vs_all / lgbm_vs_base : LightGBM vs each baseline; it is used only if BOTH exceed",
          f"{MIN_SKILL:.0%}\n")
    print(skill.round(3).to_string(index=False))
    for v in TARGET_VARS:
        s = skill[skill["var"] == v]
        print(f"\n{v}: LightGBM kept at {int(s['use_lgbm'].sum())} of {len(s)} leads")

    # ---------- 3) 80% ranges from the walk-forward errors of whichever method is used at each lead
    res = res.merge(skill[["var", "lead", "use_lgbm"]], on=["var", "lead"])
    res["chosen"] = np.where(res["use_lgbm"], res["lgbm"], res["b1"])
    res["err"] = res["y_true"] - res["chosen"]
    res["tgt_month"] = res["tgt_date"].dt.month
    q = res.groupby(["var", "tgt_month"])["err"].quantile([0.1, 0.9]).unstack()
    q.columns = ["q10", "q90"]

    # ---------- 4) final models on ALL data, forecast from the last observed month
    end = last + pd.DateOffset(months=1)
    base = fit_baseline(m, end, spec)
    base_all = fit_baseline(m, end, {})
    an = anomalies(m, base)
    feats = origin_features(an, extra)

    print("\nWHERE THINGS STAND: last 6 months vs the usual value (what the model sees right now)")
    recent = an[TARGET_VARS].tail(6).copy()
    recent.index = recent.index.strftime("%Y-%m")
    print(recent.round(3).to_string())

    pieces = []
    for v in TARGET_VARS:
        S = stack(an, feats, v, args.horizon)
        cols = feat_cols(S)
        train = S[S["y_anom"].notna()]
        model = lgb.LGBMRegressor(**LGB_PARAMS).fit(train[cols], train["y_anom"])

        P = S[S.index == last]                                  # origin = last observed month
        td = pd.DatetimeIndex(P["tgt_date"])
        b1 = base_value(base, v, td)
        lgbm = b1 + model.predict(P[cols])
        use = (skill[skill["var"] == v].set_index("lead")["use_lgbm"]
               .reindex(P["lead"]).fillna(False).astype(bool).to_numpy())
        point = np.where(use, lgbm, b1)
        qq = q.loc[v].reindex(td.month)
        lo, hi = point + qq["q10"].to_numpy(), point + qq["q90"].to_numpy()
        if v in BOUNDS:                                         # physical limits
            bmin, bmax = BOUNDS[v]
            point, lo, hi = (np.clip(a, bmin, bmax) for a in (point, lo, hi))
        pieces.append(pd.DataFrame({
            "origin": last, "target_date": td, "lead": P["lead"].to_numpy(), "variable": v,
            "method": np.where(use, "lightgbm", "baseline: " + label(spec[v])),
            "point": point, "lo10": lo, "hi90": hi,
            "baseline": b1, "all_years_mean": base_value(base_all, v, td),
        }))
    fc = pd.concat(pieces, ignore_index=True)
    fc.to_csv(proc / f"forecast_{args.lat}_{args.lon}{sfx}.csv", index=False)

    print(f"\nFORECAST from {last.date()}  (80% range from walk-forward errors)")
    show = fc.copy()
    show["target_date"] = show["target_date"].dt.strftime("%Y-%m")
    print(show.drop(columns="origin").round(3).to_string(index=False))

    outside = fc[(fc["point"] < fc["lo10"] - 1e-9) | (fc["point"] > fc["hi90"] + 1e-9)]
    if len(outside):
        print(f"\nWARNING: for {len(outside)} rows the point forecast lies outside its own 80% range.")
        print("  That means recent errors were one-sided in those months, i.e. the baseline is still biased there.")
        print("  Rows:", ", ".join(f"{r.variable}@{r.target_date:%Y-%m}" for r in outside.itertuples()))
    print(f"\nsaved: {proc / f'skill_{args.lat}_{args.lon}{sfx}.csv'}\n       {proc / f'forecast_{args.lat}_{args.lon}{sfx}.csv'}")


if __name__ == "__main__":
    main()
