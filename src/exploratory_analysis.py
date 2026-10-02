"""Produce the first EDA record and figures for PVDAQ System 1433.

Run from the project directory after the quality audit:
    python src/exploratory_analysis.py

The script reads the final master CSV and writes derived reports/figures only.
It never overwrites raw or final-source data.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent
SOURCE_CSV = WORKSPACE_DIR / "PVDAQ_1433_NASA_ERA5_900hPa_Final_Hourly.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
REPORT_DIR = PROJECT_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
TARGET = "ac_power__5069"

# Weather columns are usable as explanatory variables.  They deliberately
# exclude direct plant readings such as `pr__5067` and `dc_voltage__5070`, which
# have large gaps and would be unavailable in a real forecast before operation.
WEATHER_FEATURES = [
    "T2M",
    "RH2M",
    "WS10M",
    "WD10M",
    "PS",
    "PRECTOTCORR",
    "ALLSKY_SFC_SW_DWN",
    "CLOUD_AMT",
    "high_cloud_cover",
    "medium_cloud_cover",
    "low_cloud_cover",
    "wind_gust_10m",
    "snowfall",
    "wind_speed_900_mb",
    "wind_direction_900_mb",
]


def save_figure(path: Path) -> None:
    """Apply one consistent presentation style to all report figures."""
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()


def markdown_table(rows: list[tuple[str, str]]) -> str:
    lines = ["| Metric | Value |", "| --- | --- |"]
    lines.extend(f"| {metric} | {value} |" for metric, value in rows)
    return "\n".join(lines)


def main() -> None:
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"Final master CSV not found: {SOURCE_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SOURCE_CSV, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["hour"] = df["timestamp"].dt.hour

    # A "Complete" row contains all four 15-minute PVDAQ readings.  We retain
    # night-time values in the quality record, but use non-negative daylight
    # rows for irradiance relationship plots.
    complete = df.loc[
        (df["data_quality"] == "Complete") & df[TARGET].notna()
    ].copy()
    daylight = complete.loc[
        (complete["poa_irradiance__5061"] > 20) & (complete[TARGET] >= 0)
    ].copy()

    indexed = df.set_index("timestamp")
    monthly = indexed.resample("ME").agg(
        mean_ac_power_kw=(TARGET, "mean"),
        available_ac_power_rows=(TARGET, "count"),
        total_rows=(TARGET, "size"),
    )
    monthly["availability_pct"] = (
        monthly["available_ac_power_rows"] / monthly["total_rows"] * 100
    )
    hourly_profile = complete.groupby("hour")[TARGET].mean()

    available_weather = [column for column in WEATHER_FEATURES if column in df.columns]
    correlations = (
        complete[[TARGET, "poa_irradiance__5061", *available_weather]]
        .corr(numeric_only=True)[TARGET]
        .drop(TARGET)
        .sort_values(key=lambda values: values.abs(), ascending=False)
    )
    weather_correlations = correlations.drop(
        labels="poa_irradiance__5061", errors="ignore"
    )

    # Figure 1: monthly average power and measurement availability.
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(
        monthly.index,
        monthly["mean_ac_power_kw"],
        color="#f59e0b",
        linewidth=1.8,
    )
    axes[0].set_title("Monthly average AC power")
    axes[0].set_ylabel("kW")
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        monthly.index,
        monthly["availability_pct"],
        color="#2563eb",
        linewidth=1.8,
    )
    axes[1].axhline(100, color="#64748b", linewidth=0.8, linestyle="--")
    axes[1].set_title("Monthly AC-power availability")
    axes[1].set_ylabel("Available rows (%)")
    axes[1].set_ylim(0, 105)
    axes[1].grid(alpha=0.25)
    save_figure(FIGURE_DIR / "01_monthly_generation_and_availability.png")

    # Figure 2: average observed generation by hour of day.
    plt.figure(figsize=(10, 4.8))
    plt.plot(hourly_profile.index, hourly_profile.values, marker="o", color="#16a34a")
    plt.xticks(range(0, 24, 2))
    plt.xlabel("Hour of timestamp")
    plt.ylabel("Mean AC power (kW)")
    plt.title("Typical daily generation profile (complete PVDAQ hours)")
    plt.grid(alpha=0.25)
    save_figure(FIGURE_DIR / "02_typical_daily_generation_profile.png")

    # Figure 3: a deterministic sample keeps a multi-year scatter readable.
    scatter_sample = daylight.sample(
        n=min(12_000, len(daylight)), random_state=42
    )
    plt.figure(figsize=(8.5, 6))
    plt.scatter(
        scatter_sample["poa_irradiance__5061"],
        scatter_sample[TARGET],
        s=8,
        alpha=0.22,
        color="#7c3aed",
        edgecolors="none",
    )
    plt.xlabel("Plane-of-array irradiance")
    plt.ylabel("AC power (kW)")
    plt.title("Daylight irradiance vs. observed AC power")
    plt.grid(alpha=0.2)
    save_figure(FIGURE_DIR / "03_irradiance_vs_ac_power.png")

    # Figure 4: rank weather signals separately from the on-site irradiance
    # sensor.  This avoids presenting the sensor as a future-weather feature.
    # `weather_correlations` is already ordered by absolute strength.  Keep the
    # strongest ten, then sort by signed value only for a readable bar layout.
    plotted_correlations = weather_correlations.head(10).sort_values()
    plt.figure(figsize=(9, 5.8))
    colors = ["#dc2626" if value < 0 else "#2563eb" for value in plotted_correlations]
    plt.barh(plotted_correlations.index, plotted_correlations.values, color=colors)
    plt.axvline(0, color="#334155", linewidth=0.8)
    plt.xlabel("Pearson correlation with AC power")
    plt.title("Top weather-feature relationships (complete PVDAQ hours)")
    plt.grid(axis="x", alpha=0.2)
    save_figure(FIGURE_DIR / "04_weather_feature_correlations.png")

    top_weather_correlations = [
        {"feature": feature, "pearson_correlation": round(float(value), 4)}
        for feature, value in weather_correlations.head(8).items()
    ]
    metrics = {
        "source_csv": str(SOURCE_CSV),
        "all_hourly_rows": int(len(df)),
        "complete_target_rows": int(len(complete)),
        "daylight_complete_rows": int(len(daylight)),
        "ac_power_availability_pct": round(
            len(complete) / len(df) * 100, 3
        ),
        "mean_complete_ac_power_kw": round(float(complete[TARGET].mean()), 4),
        "max_complete_ac_power_kw": round(float(complete[TARGET].max()), 4),
        "poa_irradiance_correlation": round(
            float(correlations["poa_irradiance__5061"]), 4
        ),
        "top_weather_correlations": top_weather_correlations,
        "model_scope_note": (
            "POA irradiance is useful for same-time diagnostic regression but "
            "must not be treated as a future-known forecast input."
        ),
    }
    metrics_path = OUTPUT_DIR / "eda_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report = f"""# Exploratory Data Analysis — PVDAQ System 1433

**Source:** `{SOURCE_CSV.name}`  
**Analysis rule:** `data_quality == Complete` and non-missing `ac_power__5069`

## Data used for analysis

{markdown_table([
    ('Hourly rows in master data', f"{len(df):,}"),
    ('Complete target observations', f"{len(complete):,} ({metrics['ac_power_availability_pct']}%)"),
    ('Daylight rows for irradiance analysis', f"{len(daylight):,}"),
    ('Mean AC power across complete rows', f"{metrics['mean_complete_ac_power_kw']} kW"),
    ('Maximum AC power', f"{metrics['max_complete_ac_power_kw']} kW"),
])}

## What the figures show

1. [Monthly generation and availability](figures/01_monthly_generation_and_availability.png) separates changes in observed generation from gaps in the PVDAQ readings.
2. [Typical daily generation profile](figures/02_typical_daily_generation_profile.png) checks whether the plant follows the expected solar-shaped day cycle.
3. [Irradiance versus AC power](figures/03_irradiance_vs_ac_power.png) is a diagnostic relationship using an on-site plant sensor.
4. [Weather-feature correlations](figures/04_weather_feature_correlations.png) ranks the external-weather relationships independently of that sensor.

## Initial evidence

- The plane-of-array irradiance / AC-power Pearson correlation is **{metrics['poa_irradiance_correlation']}** across complete hours. This is expected physically, but it is a same-time measurement—not a future-known feature.
- The strongest external-weather correlations are shown below. Correlation identifies association only; it does not prove causality or choose the final model.

{markdown_table([(item['feature'], str(item['pearson_correlation'])) for item in top_weather_correlations])}

## Modelling guardrails

- Missing target rows are excluded from training and preserved in the source data.
- `dc_voltage__5070` and `pr__5067` are not baseline model features because their missingness is substantial and they are operational measurements, not weather forecasts.
- We will use chronological, not random, train/test splits. That better reflects a real prediction scenario.
- A same-timestamp model will be called **regression/nowcasting**. A future forecast will need lag features and weather known before the prediction time.
"""
    report_path = REPORT_DIR / "02_exploratory_data_analysis.md"
    report_path.write_text(report, encoding="utf-8")

    print("EDA complete")
    print(f"Complete target rows: {len(complete):,}")
    print(f"Daylight analysis rows: {len(daylight):,}")
    print(f"Report: {report_path}")
    print(f"Figures: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
