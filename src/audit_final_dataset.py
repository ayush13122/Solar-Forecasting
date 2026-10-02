"""Create a non-destructive quality audit for the final PVDAQ master dataset.

Run from the project directory:
    python src/audit_final_dataset.py

The source CSV is never edited.  The script only writes a JSON audit record and
a human-readable Markdown summary, so every later EDA or modelling step starts
from the same verified source.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent
SOURCE_CSV = WORKSPACE_DIR / "PVDAQ_1433_NASA_ERA5_900hPa_Final_Hourly.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
REPORT_DIR = PROJECT_DIR / "reports"

EXPECTED_ROWS = 61_726
EXPECTED_COLUMNS = [
    "timestamp",
    "ac_power__5069",
    "ambient_temp__5062",
    "module_temp__5063",
    "poa_irradiance__5061",
    "dc_voltage__5070",
    "pr__5067",
    "kwh_gross__5065",
    "reading_count",
    "T2M",
    "RH2M",
    "WS10M",
    "WD10M",
    "PS",
    "PRECTOTCORR",
    "data_quality",
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


def percentage(part: int, total: int) -> float:
    """Return a rounded percentage while safely handling an empty dataset."""
    return round((part / total) * 100, 3) if total else 0.0


def markdown_table(rows: list[tuple[str, str]]) -> str:
    """Build a tiny Markdown table without another reporting dependency."""
    lines = ["| Check | Result |", "| --- | --- |"]
    lines.extend(f"| {label} | {value} |" for label, value in rows)
    return "\n".join(lines)


def main() -> None:
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(
            "Final master CSV was not found at "
            f"{SOURCE_CSV}. Keep the source file in the workspace root."
        )

    df = pd.read_csv(SOURCE_CSV, parse_dates=["timestamp"])
    row_count, column_count = df.shape

    actual_columns = df.columns.tolist()
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in actual_columns]
    unexpected_columns = [column for column in actual_columns if column not in EXPECTED_COLUMNS]
    duplicate_timestamps = int(df["timestamp"].duplicated().sum())
    timestamps = df["timestamp"].dropna().sort_values()

    if timestamps.empty:
        expected_hourly_index = pd.DatetimeIndex([])
        missing_timestamps = pd.DatetimeIndex([])
    else:
        expected_hourly_index = pd.date_range(
            start=timestamps.iloc[0], end=timestamps.iloc[-1], freq="h"
        )
        missing_timestamps = expected_hourly_index.difference(pd.DatetimeIndex(timestamps))

    missing_values = {
        column: {
            "count": int(df[column].isna().sum()),
            "percent": percentage(int(df[column].isna().sum()), row_count),
        }
        for column in actual_columns
    }
    quality_counts = {
        str(label): int(count)
        for label, count in df["data_quality"].value_counts(dropna=False).items()
    }
    reading_counts = {
        str(label): int(count)
        for label, count in df["reading_count"].value_counts(dropna=False).sort_index().items()
    }

    target = df["ac_power__5069"]
    valid_target = target.dropna()
    target_summary = {
        "available_rows": int(valid_target.size),
        "missing_rows": int(target.isna().sum()),
        "zero_or_negative_rows": int((valid_target <= 0).sum()),
        "negative_rows": int((valid_target < 0).sum()),
        "minimum_kw": round(float(valid_target.min()), 4),
        "maximum_kw": round(float(valid_target.max()), 4),
        "mean_kw": round(float(valid_target.mean()), 4),
        "median_kw": round(float(valid_target.median()), 4),
    }

    validation = {
        "expected_shape": row_count == EXPECTED_ROWS and column_count == len(EXPECTED_COLUMNS),
        "expected_schema": not missing_columns and not unexpected_columns,
        "unique_timestamps": duplicate_timestamps == 0,
        "usable_target_observations": target_summary["available_rows"] > 0,
    }
    ready_for_eda = all(validation.values())

    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(SOURCE_CSV),
        "source_file_bytes": SOURCE_CSV.stat().st_size,
        "shape": {"rows": row_count, "columns": column_count},
        "expected_shape": {"rows": EXPECTED_ROWS, "columns": len(EXPECTED_COLUMNS)},
        "timestamp_range": {
            "start": timestamps.iloc[0].isoformat() if not timestamps.empty else None,
            "end": timestamps.iloc[-1].isoformat() if not timestamps.empty else None,
            "duplicate_count": duplicate_timestamps,
            "missing_hour_count": int(len(missing_timestamps)),
            "first_missing_hours": [
                value.isoformat() for value in missing_timestamps[:10]
            ],
        },
        "schema": {
            "actual_columns": actual_columns,
            "missing_expected_columns": missing_columns,
            "unexpected_columns": unexpected_columns,
        },
        "data_quality_counts": quality_counts,
        "reading_count_distribution": reading_counts,
        "missing_values": missing_values,
        "target_ac_power_kw": target_summary,
        "validation": validation,
        "ready_for_eda": ready_for_eda,
        "note": (
            "This is a non-destructive audit. Missing values and zero/negative "
            "night-time readings are reported, not imputed or removed."
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "final_dataset_quality_report.json"
    markdown_path = REPORT_DIR / "01_data_quality_audit.md"
    json_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    missing_rows = [
        (
            column,
            f"{details['count']:,} ({details['percent']:.3f}%)",
        )
        for column, details in missing_values.items()
        if details["count"]
    ]
    if not missing_rows:
        missing_rows = [("None", "No missing values found")]

    report = f"""# Final Dataset Quality Audit

**Status:** {'Ready for EDA' if ready_for_eda else 'Review required'}  
**Source:** `{SOURCE_CSV.name}`  
**Audit generated (UTC):** {audit['generated_at_utc']}

## Dataset identity

{markdown_table([
    ('Rows', f'{row_count:,} (expected {EXPECTED_ROWS:,})'),
    ('Columns', f'{column_count} (expected {len(EXPECTED_COLUMNS)})'),
    ('Time range', f"{audit['timestamp_range']['start']} to {audit['timestamp_range']['end']}"),
    ('Duplicate timestamps', f'{duplicate_timestamps:,}'),
    ('Missing hourly timestamps', f"{len(missing_timestamps):,}"),
    ('Schema differences', f"{len(missing_columns)} missing, {len(unexpected_columns)} unexpected"),
])}

## PVDAQ target: `ac_power__5069`

{markdown_table([
    ('Available observations', f"{target_summary['available_rows']:,}"),
    ('Missing observations', f"{target_summary['missing_rows']:,}"),
    ('Minimum / maximum kW', f"{target_summary['minimum_kw']} / {target_summary['maximum_kw']}"),
    ('Mean / median kW', f"{target_summary['mean_kw']} / {target_summary['median_kw']}"),
    ('Zero or negative observations', f"{target_summary['zero_or_negative_rows']:,}"),
    ('Negative observations', f"{target_summary['negative_rows']:,}"),
])}

Zero and slightly negative generation values are retained for now; they commonly
occur at night or can reflect sensor noise. We will make any modelling filter
explicit in the next step instead of silently changing the source data.

## `data_quality` labels

{markdown_table([(label, f'{count:,}') for label, count in quality_counts.items()])}

## Missing values by column

{markdown_table(missing_rows)}

## Next approved project step

Perform exploratory data analysis from this verified master CSV: inspect power
generation over time, sunlight/irradiance relationships, weather correlations,
and missing-data patterns. The raw PVDAQ and final master CSVs remain unchanged.
"""
    markdown_path.write_text(report, encoding="utf-8")

    print("Final master dataset audit complete")
    print(f"Source: {SOURCE_CSV}")
    print(f"Shape: {row_count:,} rows x {column_count} columns")
    print(f"Ready for EDA: {ready_for_eda}")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")


if __name__ == "__main__":
    main()
