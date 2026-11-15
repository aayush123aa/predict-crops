import argparse
import calendar
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def trapezoid_score(val: float, vmin: float, vopmn: float, vopmx: float, vmax: float) -> float:
    if val is None or math.isnan(val):
        return 0.5
    if val < vmin or val > vmax:
        return 0.0
    if vopmn <= val <= vopmx:
        return 1.0
    if val < vopmn:
        return max(0.0, (val - vmin) / max(1e-6, (vopmn - vmin)))
    return max(0.0, (vmax - val) / max(1e-6, (vmax - vopmx)))


def load_soil_terrain(soil_path: Path | None, lat: float, lon: float) -> dict:
    candidates = []
    if soil_path is not None:
        candidates.append(soil_path)
    candidates.extend([
        Path(f"data/processed/soil_terrain_{lat}_{lon}.json"),
        Path(f"data/processed/soil_{lat}_{lon}.json"),
    ])

    for p in candidates:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return {
                "ph": float(raw.get("ph", raw.get("phh2o", 6.5))),
                "clay_pct": float(raw.get("clay_pct", raw.get("clay", 25.0))),
                "sand_pct": float(raw.get("sand_pct", raw.get("sand", 40.0))),
                "silt_pct": float(raw.get("silt_pct", raw.get("silt", 35.0))),
                "soc_g_kg": float(raw.get("soc_g_kg", raw.get("soc", 15.0))),
                "nitrogen_g_kg": float(raw.get("nitrogen_g_kg", raw.get("nitrogen", 1.4))),
                "elevation_m": float(raw.get("elevation_m", raw.get("elevation", 80.0))),
                "slope_pct": float(raw.get("slope_pct", raw.get("slope", 1.5))),
                "source": str(p),
            }

    return {
        "ph": 6.5,
        "clay_pct": 25.0,
        "sand_pct": 40.0,
        "silt_pct": 35.0,
        "soc_g_kg": 15.0,
        "nitrogen_g_kg": 1.4,
        "elevation_m": 80.0,
        "slope_pct": 1.5,
        "source": "neutral_global_fallback",
    }


def load_crops_catalog(crops_path: Path | None) -> pd.DataFrame:
    candidates = []
    if crops_path is not None:
        candidates.append(crops_path)
    candidates.extend([
        Path("data/processed/crops_prepared.csv"),
        Path("data/crops_prepared.csv"),
        Path("crops_prepared.csv"),
    ])

    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            if "crop_id" in df.columns and "sow_tmin" in df.columns:
                return df

    # Auto-build via prepare_crops if missing
    from prepare_crops import build_prepared_crops
    df = build_prepared_crops()
    out_default = Path("data/processed/crops_prepared.csv")
    out_default.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_default, index=False)
    return df


def build_36_month_timeline(
    forecast_path: Path,
    delta_temp: float = 0.0,
    delta_rain_pct: float = 0.0,
) -> list[dict]:
    if not forecast_path.exists():
        raise FileNotFoundError(
            f"Forecast file not found: {forecast_path}. Run forecast_model.py first."
        )

    df = pd.read_csv(forecast_path)
    if "method" in df.columns:
        preferred = df[df["method"].isin(["hybrid", "lgbm", "ml"])]
        if not preferred.empty:
            df = preferred

    df["target_date"] = pd.to_datetime(df["target_date"])
    df = df.sort_values(["target_date", "variable"]).drop_duplicates(
        subset=["target_date", "variable"], keep="first"
    )

    dates = sorted(df["target_date"].unique())[:12]
    if len(dates) < 12:
        raise ValueError(f"Expected 12 forecast months in {forecast_path}, found {len(dates)}.")

    base_12 = []
    for dt in dates:
        sub = df[df["target_date"] == dt].set_index("variable")
        t_pt = float(sub.loc["temp_mean", "point"]) if "temp_mean" in sub.index else 15.0
        r_pt = float(sub.loc["rain_mm", "point"]) if "rain_mm" in sub.index else 60.0
        r_lo = float(sub.loc["rain_mm", "lo10"]) if "rain_mm" in sub.index and "lo10" in sub.columns else r_pt * 0.7
        sw_pt = float(sub.loc["soilw_root", "point"]) if "soilw_root" in sub.index else 0.55
        sw_lo = float(sub.loc["soilw_root", "lo10"]) if "soilw_root" in sub.index and "lo10" in sub.columns else sw_pt * 0.85

        base_12.append({
            "cal_month": int(pd.Timestamp(dt).month),
            "cal_year": int(pd.Timestamp(dt).year),
            "temp": t_pt,
            "rain": max(0.0, r_pt),
            "rain_lo10": max(0.0, r_lo),
            "soilw": float(np.clip(sw_pt, 0.05, 1.0)),
            "soilw_lo10": float(np.clip(sw_lo, 0.05, 1.0)),
        })

    timeline = []
    for t in range(38):  # 36 months + 2 months grace for late-horizon harvest evaluation
        b = base_12[t % 12]
        yr_idx = t // 12
        t_shift = delta_temp if yr_idx >= 1 else 0.0
        r_mult = (1.0 + delta_rain_pct / 100.0) if yr_idx >= 1 else 1.0
        sw_shift = (delta_rain_pct / 100.0) * 0.12 if yr_idx >= 1 else 0.0

        timeline.append({
            "t": t,
            "year_num": yr_idx + 1,
            "cal_month": b["cal_month"],
            "cal_year": b["cal_year"] + yr_idx,
            "temp": b["temp"] + t_shift,
            "rain": max(0.0, b["rain"] * r_mult),
            "rain_lo10": max(0.0, b["rain_lo10"] * r_mult),
            "soilw": float(np.clip(b["soilw"] + sw_shift, 0.05, 1.0)),
            "soilw_lo10": float(np.clip(b["soilw_lo10"] + sw_shift, 0.05, 1.0)),
        })
    return timeline


def evaluate_crop_window(
    crop: pd.Series,
    start_t: int,
    timeline: list[dict],
    soil: dict,
    irrigation: str,
) -> tuple[float, float]:
    """
    Returns (biophysical_fit [0..1], drought_resilience [0..1]) for planting `crop` at month `start_t`.
    Returns (0.0, 0.0) if any physiological or establishment gate fails.
    """
    c_len = int(crop["cycle_months"])
    if start_t + c_len > len(timeline):
        return 0.0, 0.0

    window = timeline[start_t : start_t + c_len]
    sow_slot = window[0]

    # 1. Sowing Thermal Gate
    if not (float(crop["sow_tmin"]) <= sow_slot["temp"] <= float(crop["sow_tmax"])):
        return 0.0, 0.0

    # 2. Seedbed Establishment Moisture Gate
    req_soilw = float(crop["min_sowing_soilw"])
    eff_sow_soilw = 0.85 if irrigation == "irrigated" else (
        max(sow_slot["soilw"], sow_slot["soilw"] + 0.08) if irrigation == "limited_supplemental" else sow_slot["soilw"]
    )
    if eff_sow_soilw < req_soilw:
        return 0.0, 0.0

    temps = [m["temp"] for m in window]

    # 3. Juvenile Vernalization Gate (chilling within first 5 months after sowing)
    if bool(crop["vernalization_required"]):
        juv_temps = temps[1 : min(6, len(temps))]
        chilling_months = sum(1 for tv in juv_temps if tv <= 8.5)
        if chilling_months < 2:
            return 0.0, 0.0

    # 4. Cool-Season Cover Gate (prevents overwinter covers from occupying peak summer)
    if bool(crop.get("cool_season_cover", False)):
        if sum(1 for tv in temps if tv <= 9.5) < 2:
            return 0.0, 0.0

    # 5. Ripening / Termination Thermal Gate (final 2 months)
    ripen_temp = float(np.mean(temps[-2:]))
    if ripen_temp < float(crop["harvest_tmin"]):
        return 0.0, 0.0

    # 6. Extreme Survival Bounds
    if min(temps) < float(crop["tmin"]) or max(temps) > float(crop["tmax"]):
        return 0.0, 0.0

    # 7. Continuous Trapezoidal Scoring (Temperature, Water, Soil pH, Edaphic Modifiers)
    t_mean = float(np.mean(temps))
    t_score = trapezoid_score(
        t_mean, float(crop["tmin"]), float(crop["topmn"]), float(crop["topmx"]), float(crop["tmax"])
    )

    rain_sum = sum(m["rain"] for m in window)
    rain_lo10_sum = sum(m["rain_lo10"] for m in window)
    if irrigation == "irrigated":
        eff_rain = max(rain_sum, float(crop["ropmn"]) * (c_len / 12.0))
        eff_rain_lo10 = eff_rain * 0.90
    elif irrigation == "limited_supplemental":
        eff_rain = rain_sum + 90.0 * (c_len / 12.0)
        eff_rain_lo10 = rain_lo10_sum + 75.0 * (c_len / 12.0)
    else:
        eff_rain = rain_sum
        eff_rain_lo10 = rain_lo10_sum

    ann_rain_eq = eff_rain * (12.0 / c_len)
    ann_rain_lo10_eq = eff_rain_lo10 * (12.0 / c_len)

    r_score = trapezoid_score(
        ann_rain_eq, float(crop["rmin"]), float(crop["ropmn"]), float(crop["ropmx"]), float(crop["rmax"])
    )
    drought_res = trapezoid_score(
        ann_rain_lo10_eq, float(crop["rmin"]), float(crop["ropmn"]), float(crop["ropmx"]), float(crop["rmax"])
    )

    ph_score = trapezoid_score(
        soil["ph"], float(crop["phmin"]), float(crop["phopmn"]), float(crop["phopmx"]), float(crop["phmax"])
    )

    # Soil drainage & slope modifier
    edaphic_mod = 1.0
    mean_soilw = float(np.mean([m["soilw"] for m in window]))
    if not bool(crop["waterlogging_ok"]) and soil["clay_pct"] > 45.0 and mean_soilw > 0.78:
        edaphic_mod *= 0.85
    if bool(crop["waterlogging_ok"]) and soil["sand_pct"] > 65.0 and irrigation == "rainfed":
        edaphic_mod *= 0.70
    if soil["slope_pct"] > 8.0 and crop["root_depth_class"] == "shallow" and not bool(crop["is_cover_crop"]):
        edaphic_mod *= 0.88

    fit = edaphic_mod * (0.42 * t_score + 0.38 * r_score + 0.20 * ph_score)
    return round(float(np.clip(fit, 0.0, 1.0)), 4), round(float(np.clip(drought_res, 0.0, 1.0)), 4)


def compute_soil_stewardship_index(events: list[dict], horizon: int = 36) -> dict:
    """
    Decomposed 0-100 Soil Stewardship Index (no uncalibrated physical mass claims):
      1. Living Cover Ratio (0-100): fraction of 36 months with active crop/cover roots
      2. Biological N Balance (0-100): legume N fixation vs heavy-feeder demand
      3. Root & Residue Input (0-100): deep taproots, cover crops & high-residue cereals
      4. Taxonomic & Root Diversity (0-100): Shannon family diversity + root-depth alternation
    """
    if not events:
        return {
            "living_cover_ratio": 0.0,
            "biological_n_balance": 0.0,
            "root_residue_input": 0.0,
            "taxonomic_root_diversity": 0.0,
            "composite_stewardship_score": 0.0,
        }

    # 1. Living Cover Ratio
    covered_months = sum(min(e["end_t"], horizon) - e["start_t"] for e in events)
    living_cover = min(100.0, (covered_months / horizon) * 100.0)

    # 2. Biological N Balance (optimal around ~30-45% fixer months; penalized if >55% legumes)
    fixer_months = sum(
        (min(e["end_t"], horizon) - e["start_t"])
        for e in events if e["nitrogen_role"] == "fixer"
    )
    fixer_share = fixer_months / max(1, covered_months)
    if fixer_share <= 0.35:
        n_balance = 35.0 + (fixer_share / 0.35) * 65.0
    elif fixer_share <= 0.55:
        n_balance = 100.0 - ((fixer_share - 0.35) / 0.20) * 15.0
    else:
        n_balance = max(40.0, 85.0 - ((fixer_share - 0.55) / 0.45) * 45.0)

    # 3. Root & Residue Input
    residue_pts = 0.0
    for e in events:
        dur = min(e["end_t"], horizon) - e["start_t"]
        w = 1.0
        if e["is_cover_crop"]:
            w += 0.65
        if e["root_depth_class"] == "deep":
            w += 0.35
        elif e["root_depth_class"] == "medium":
            w += 0.15
        residue_pts += dur * w
    root_residue = min(100.0, (residue_pts / (horizon * 1.45)) * 100.0)

    # 4. Taxonomic & Root Diversity (Shannon diversity over families + root depth transitions)
    fam_counts = Counter(e["family"] for e in events)
    n_ev = len(events)
    if n_ev <= 1 or len(fam_counts) <= 1:
        shannon_norm = 0.0
    else:
        h = -sum((c / n_ev) * math.log(c / n_ev) for c in fam_counts.values())
        shannon_norm = min(1.0, h / math.log(min(4, n_ev)))

    root_transitions = sum(
        1 for i in range(1, n_ev)
        if events[i]["root_depth_class"] != events[i - 1]["root_depth_class"]
    )
    trans_ratio = root_transitions / max(1, n_ev - 1) if n_ev > 1 else 0.0
    diversity = (0.65 * shannon_norm + 0.35 * trans_ratio) * 100.0

    composite = (
        0.30 * living_cover
        + 0.25 * n_balance
        + 0.25 * root_residue
        + 0.20 * diversity
    )

    return {
        "living_cover_ratio": round(living_cover, 1),
        "biological_n_balance": round(n_balance, 1),
        "root_residue_input": round(root_residue, 1),
        "taxonomic_root_diversity": round(diversity, 1),
        "composite_stewardship_score": round(composite, 1),
    }


def optimize_rotation_beam(
    crops_df: pd.DataFrame,
    timeline: list[dict],
    soil: dict,
    irrigation: str,
    strategy: str,
    horizon: int = 36,
    beam_width: int = 45,
) -> dict:
    # Precompute viable (crop_idx, start_t) -> (fit, drought_res)
    eval_cache: dict[tuple[int, int], tuple[float, float]] = {}
    for idx, crop in crops_df.iterrows():
        for t in range(horizon):
            fit, dr = evaluate_crop_window(crop, t, timeline, soil, irrigation)
            if fit > 0.05:
                eval_cache[(idx, t)] = (fit, dr)

    beam = [(0.0, 0, [])]
    completed = []

    while beam:
        next_beam = []
        for cum_score, t, events in beam:
            if t >= horizon:
                completed.append((cum_score, events))
                continue

            # Transition option: 1-month inter-crop stubble/seedbed window
            consec_idle = (t - events[-1]["end_t"]) if events else t
            idle_pen = -0.14 if strategy == "current_baseline" else -0.10
            if consec_idle >= 4:
                idle_pen -= 0.35
            next_beam.append((cum_score + idle_pen, t + 1, events))

            for idx, crop in crops_df.iterrows():
                if (idx, t) not in eval_cache:
                    continue
                c_len = int(crop["cycle_months"])
                if t + c_len > horizon + 2:
                    continue

                is_cover = bool(crop["is_cover_crop"])
                # Baseline reflects cash-crop-only business as usual
                if strategy == "current_baseline" and is_cover:
                    continue
                # Never stack two cover crops consecutively without a cash crop
                if is_cover and events and events[-1]["is_cover_crop"]:
                    continue

                fit, drought_res = eval_cache[(idx, t)]
                adj = 0.0

                if strategy == "current_baseline":
                    # Conventional baseline avoids immediate same-species monoculture
                    if events and events[-1]["crop_id"] == crop["crop_id"]:
                        adj -= 0.30
                else:
                    # Multi-year pathology & taxonomic cooldowns
                    for prev in reversed(events):
                        months_since = t - prev["end_t"]
                        if crop["family"] == "Fabaceae" and prev["family"] == "Fabaceae" and months_since < 48:
                            adj -= 0.50 if not is_cover else 0.25
                        if crop["family"] == "Brassicaceae" and prev["family"] == "Brassicaceae" and months_since < 36:
                            adj -= 0.45
                        if crop["crop_id"] == prev["crop_id"] and months_since < 24:
                            adj -= 0.35

                    if events:
                        last = events[-1]
                        if crop["family"] == "Poaceae" and last["family"] == "Poaceae":
                            # Oats (Avena) produce avenacin that suppresses wheat take-all
                            adj -= 0.02 if last["genus"] == "Avena" else 0.14
                        if crop["nitrogen_role"] == "heavy_feeder" and last["nitrogen_role"] == "fixer":
                            adj += 0.10
                        if crop["root_depth_class"] != last["root_depth_class"]:
                            adj += 0.06
                        if crop["family"] != last["family"]:
                            adj += 0.08

                    if strategy == "soil_regenerative":
                        if is_cover:
                            adj += 0.16
                        if crop["nitrogen_role"] == "fixer":
                            adj += 0.09
                        if crop["root_depth_class"] == "deep":
                            adj += 0.05
                    elif strategy == "water_resilient":
                        # Prioritize drought resilience, low water demand & deep rooting
                        adj += 0.22 * (drought_res - fit)
                        if float(crop["ropmn"]) <= 480.0:
                            adj += 0.10
                        if crop["root_depth_class"] == "deep":
                            adj += 0.08
                        if is_cover:
                            adj += 0.06

                eff_months = min(t + c_len, horizon) - t
                step_gain = eff_months * (fit + adj)
                sow_slot = timeline[t]
                end_slot = timeline[min(t + c_len, len(timeline) - 1)]

                ev = {
                    "crop_id": str(crop["crop_id"]),
                    "crop_name": str(crop["crop_name"]),
                    "scientific_name": str(crop["scientific_name"]),
                    "genus": str(crop["genus"]),
                    "family": str(crop["family"]),
                    "nitrogen_role": str(crop["nitrogen_role"]),
                    "root_depth_class": str(crop["root_depth_class"]),
                    "is_cover_crop": is_cover,
                    "start_t": t,
                    "end_t": t + c_len,
                    "cycle_months": c_len,
                    "year_num": (t // 12) + 1,
                    "sow_label": f"{calendar.month_abbr[sow_slot['cal_month']]} Y{(t // 12) + 1}",
                    "harvest_label": f"{calendar.month_abbr[end_slot['cal_month']]} Y{((t + c_len - 1) // 12) + 1}",
                    "biophysical_fit": fit,
                    "drought_resilience": drought_res,
                }
                next_beam.append((cum_score + step_gain, t + c_len, events + [ev]))

        next_beam.sort(key=lambda x: x[0], reverse=True)
        beam = next_beam[:beam_width]

    completed.sort(key=lambda x: x[0], reverse=True)
    best_score, best_events = completed[0] if completed else (0.0, [])

    if best_events:
        weights = [min(e["end_t"], horizon) - e["start_t"] for e in best_events]
        mean_fit = float(np.average([e["biophysical_fit"] for e in best_events], weights=weights))
        mean_drought = float(np.average([e["drought_resilience"] for e in best_events], weights=weights))
    else:
        mean_fit, mean_drought = 0.0, 0.0

    stewardship = compute_soil_stewardship_index(best_events, horizon=horizon)

    return {
        "strategy": strategy,
        "mean_biophysical_fit": round(mean_fit, 3),
        "mean_drought_resilience": round(mean_drought, 3),
        "soil_stewardship": stewardship,
        "schedule": best_events,
    }


def export_monthly_crop_scores(
    crops_df: pd.DataFrame,
    timeline: list[dict],
    soil: dict,
    irrigation: str,
) -> pd.DataFrame:
    rows = []
    for _, crop in crops_df.iterrows():
        for t in range(12):
            fit, dr = evaluate_crop_window(crop, t, timeline, soil, irrigation)
            if fit > 0.0:
                m_num = timeline[t]["cal_month"]
                rows.append({
                    "crop_id": crop["crop_id"],
                    "crop_name": crop["crop_name"],
                    "family": crop["family"],
                    "sow_month_num": m_num,
                    "sow_month": calendar.month_abbr[m_num],
                    "cycle_months": int(crop["cycle_months"]),
                    "is_cover_crop": bool(crop["is_cover_crop"]),
                    "biophysical_fit": fit,
                    "drought_resilience": dr,
                })
    return pd.DataFrame(rows).sort_values(["sow_month_num", "biophysical_fit"], ascending=[True, False])


def main() -> None:
    parser = argparse.ArgumentParser(description="Global Climate-Gated 3-Year Crop Rotation Optimizer")
    parser.add_argument("--lat", type=float, required=True, help="Latitude")
    parser.add_argument("--lon", type=float, required=True, help="Longitude")
    parser.add_argument(
        "--irrigation",
        type=str,
        default="limited_supplemental",
        choices=["rainfed", "limited_supplemental", "irrigated"],
        help="Water management regime",
    )
    parser.add_argument("--delta-temp", type=float, default=0.0, help="Year 2-3 warming perturbation (°C)")
    parser.add_argument("--delta-rain-pct", type=float, default=0.0, help="Year 2-3 rainfall perturbation (%%)")
    parser.add_argument("--soil", type=Path, default=None, help="Path to soil_terrain JSON")
    parser.add_argument("--crops", type=Path, default=None, help="Path to crops_prepared.csv")
    parser.add_argument("--forecast", type=Path, default=None, help="Path to forecast_{lat}_{lon}.csv")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"), help="Output directory")
    args = parser.parse_args()

    forecast_path = args.forecast or (args.out_dir / f"forecast_{args.lat}_{args.lon}.csv")
    soil = load_soil_terrain(args.soil, args.lat, args.lon)
    crops_df = load_crops_catalog(args.crops)
    timeline = build_36_month_timeline(forecast_path, args.delta_temp, args.delta_rain_pct)

    strategies = {}
    for strat in ("current_baseline", "soil_regenerative", "water_resilient"):
        strategies[strat] = optimize_rotation_beam(
            crops_df, timeline, soil, args.irrigation, strategy=strat
        )

    scores_df = export_monthly_crop_scores(crops_df, timeline, soil, args.irrigation)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_out = args.out_dir / f"rotation_recommendations_{args.lat}_{args.lon}.json"
    csv_out = args.out_dir / f"crop_scores_{args.lat}_{args.lon}.csv"

    report = {
        "coordinates": {"lat": args.lat, "lon": args.lon},
        "scenario": {
            "irrigation": args.irrigation,
            "delta_temp_c": args.delta_temp,
            "delta_rain_pct": args.delta_rain_pct,
        },
        "soil_terrain": soil,
        "year1_monthly_outlook": timeline[:12],
        "strategies": strategies,
    }
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    scores_df.to_csv(csv_out, index=False)

    # Terminal Summary
    print(f"\n=== Crop Rotation Recommendations ({args.lat}°N, {args.lon}°E | {args.irrigation}) ===")
    print(
        f"Soil ({soil['source']}): pH {soil['ph']:.1f} | "
        f"Clay {soil['clay_pct']:.1f}% / Sand {soil['sand_pct']:.1f}% | "
        f"SOC {soil['soc_g_kg']:.1f} g/kg | Slope {soil['slope_pct']:.1f}%"
    )

    y1 = timeline[:12]
    print(
        f"Year 1 Outlook: Mean Temp {np.mean([m['temp'] for m in y1]):.2f}°C "
        f"({min(m['temp'] for m in y1):.1f}–{max(m['temp'] for m in y1):.1f}°C) | "
        f"Annual Rain {sum(m['rain'] for m in y1):.1f} mm | "
        f"Mean Root SoilW {np.mean([m['soilw'] for m in y1]):.2f}"
    )

    labels = {
        "current_baseline": "Current Baseline (Emergent Biophysical)",
        "soil_regenerative": "Soil-Regenerative",
        "water_resilient": "Water-Resilient",
    }
    for key, title in labels.items():
        res = strategies[key]
        st = res["soil_stewardship"]
        print(
            f"\n--- Strategy: {title} "
            f"(Fit: {res['mean_biophysical_fit']:.3f} | "
            f"Drought: {res['mean_drought_resilience']:.3f} | "
            f"Stewardship: {st['composite_stewardship_score']:.1f}/100) ---"
        )
        print(
            f"    Stewardship Breakdown -> Cover: {st['living_cover_ratio']:.0f}% | "
            f"N-Balance: {st['biological_n_balance']:.0f}/100 | "
            f"Root/Residue: {st['root_residue_input']:.0f}/100 | "
            f"Diversity: {st['taxonomic_root_diversity']:.0f}/100"
        )
        for ev in res["schedule"]:
            tag = " [Cover]" if ev["is_cover_crop"] else ""
            print(
                f"  Y{ev['year_num']} [{ev['sow_label']} -> {ev['harvest_label']}, {ev['cycle_months']}m]: "
                f"{ev['crop_name']}{tag} ({ev['family']}, fit={ev['biophysical_fit']:.2f})"
            )

    print(f"\nSaved JSON report -> {json_out}")
    print(f"Saved crop scores -> {csv_out}")


if __name__ == "__main__":
    main()
