from pathlib import Path
import json
import pandas as pd
import numpy as np

# ============================================================
# HELPER FUNCTION
# ============================================================

def safe_float(value):
    if pd.isna(value):
        return None

    return float(value)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_FILE = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "PVDAQ_1433_NASA_ERA5_900hPa_Final_Hourly.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_FILE = REPORT_DIR / "master_dataset_audit.json"


# ============================================================
# EXPECTED COLUMNS
# ============================================================

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
    "wind_direction_900_mb"
]


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("MASTER DATASET AUDIT")
print("=" * 70)

print("\nLoading dataset...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nDataset not found:\n{INPUT_FILE}\n\n"
        "Check that the final CSV is inside data/processed/"
    )

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded successfully.")


# ============================================================
# BASIC STRUCTURE
# ============================================================

print("\n" + "=" * 70)
print("1. BASIC DATASET STRUCTURE")
print("=" * 70)

rows, columns = df.shape

print(f"Rows    : {rows:,}")
print(f"Columns : {columns}")

print("\nColumn names:")
for i, col in enumerate(df.columns, start=1):
    print(f"{i:02d}. {col}")


# ============================================================
# DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("2. DATA TYPES")
print("=" * 70)

dtype_info = {}

for col in df.columns:
    dtype_info[col] = str(df[col].dtype)
    print(f"{col:<35} {df[col].dtype}")


# ============================================================
# SCHEMA VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("3. SCHEMA VALIDATION")
print("=" * 70)

missing_columns = [
    col for col in EXPECTED_COLUMNS
    if col not in df.columns
]

unexpected_columns = [
    col for col in df.columns
    if col not in EXPECTED_COLUMNS
]

schema_match = (
        df.columns.tolist() == EXPECTED_COLUMNS
)

print(f"Expected column count : {len(EXPECTED_COLUMNS)}")
print(f"Actual column count   : {len(df.columns)}")
print(f"Schema exact match    : {schema_match}")

if missing_columns:
    print("\nMissing expected columns:")
    for col in missing_columns:
        print(" -", col)

if unexpected_columns:
    print("\nUnexpected columns:")
    for col in unexpected_columns:
        print(" -", col)


# ============================================================
# TIMESTAMP AUDIT
# ============================================================

print("\n" + "=" * 70)
print("4. TIMESTAMP AUDIT")
print("=" * 70)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

invalid_timestamps = int(df["timestamp"].isna().sum())

valid_timestamps = df["timestamp"].dropna()

if len(valid_timestamps) > 0:

    start_timestamp = valid_timestamps.min()
    end_timestamp = valid_timestamps.max()

    print(f"Start timestamp       : {start_timestamp}")
    print(f"End timestamp         : {end_timestamp}")

else:
    start_timestamp = None
    end_timestamp = None

    print("No valid timestamps found.")


# ============================================================
# DUPLICATE TIMESTAMP CHECK
# ============================================================

duplicate_timestamp_count = int(
    df["timestamp"].duplicated().sum()
)

print(f"Invalid timestamps    : {invalid_timestamps}")
print(f"Duplicate timestamps  : {duplicate_timestamp_count}")


# ============================================================
# HOURLY TIMESTAMP CONTINUITY
# ============================================================

print("\nChecking hourly timestamp continuity...")

sorted_timestamps = (
    df["timestamp"]
    .dropna()
    .sort_values()
)

if len(sorted_timestamps) > 1:

    time_diff = sorted_timestamps.diff()

    expected_interval = pd.Timedelta(hours=1)

    gap_mask = time_diff > expected_interval

    missing_hour_count = int(
        ((time_diff[gap_mask] / expected_interval) - 1)
        .sum()
    )

    largest_gap = time_diff.max()

else:

    missing_hour_count = None
    largest_gap = None


print(f"Missing hourly timestamps : {missing_hour_count}")
print(f"Largest timestamp gap     : {largest_gap}")


# ============================================================
# MISSING VALUE AUDIT
# ============================================================

print("\n" + "=" * 70)
print("5. MISSING VALUE AUDIT")
print("=" * 70)

missing_count = df.isna().sum()

missing_percentage = (
        df.isna().mean() * 100
)

missing_report = {}

for col in df.columns:

    count = int(missing_count[col])
    percentage = float(missing_percentage[col])

    missing_report[col] = {
        "missing_count": count,
        "missing_percentage": round(percentage, 3)
    }

    print(
        f"{col:<35} "
        f"{count:>7,} missing "
        f"({percentage:>6.2f}%)"
    )


# ============================================================
# READING COUNT / DATA QUALITY
# ============================================================

print("\n" + "=" * 70)
print("6. DATA QUALITY AUDIT")
print("=" * 70)

reading_count_distribution = {}

if "reading_count" in df.columns:

    reading_count_distribution = (
        df["reading_count"]
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )

    print("\nReading count distribution:")

    for value, count in reading_count_distribution.items():
        print(f"  {value} : {count:,}")


data_quality_distribution = {}

if "data_quality" in df.columns:

    data_quality_distribution = (
        df["data_quality"]
        .value_counts(dropna=False)
        .to_dict()
    )

    print("\nData quality distribution:")

    for value, count in data_quality_distribution.items():
        print(f"  {value} : {count:,}")


# ============================================================
# NUMERIC COLUMN STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("7. NUMERIC FEATURE SUMMARY")
print("=" * 70)

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

numeric_summary = {}

for col in numeric_columns:

    series = df[col]

    numeric_summary[col] = {
        "min": safe_float(series.min()),
        "max": safe_float(series.max()),
        "mean": safe_float(series.mean()),
        "median": safe_float(series.median()),
        "std": safe_float(series.std())
    }

    print(
        f"{col:<30} "
        f"min={series.min():.4f} | "
        f"max={series.max():.4f} | "
        f"mean={series.mean():.4f}"
    )


# ============================================================
# TARGET VARIABLE AUDIT
# ============================================================

print("\n" + "=" * 70)
print("8. TARGET VARIABLE AUDIT")
print("=" * 70)

target = "ac_power__5069"

target_report = {}

if target in df.columns:

    target_series = df[target]

    target_missing = int(target_series.isna().sum())
    target_available = int(target_series.notna().sum())

    zero_count = int(
        (target_series == 0).sum()
    )

    negative_count = int(
        (target_series < 0).sum()
    )

    positive_count = int(
        (target_series > 0).sum()
    )

    target_report = {
        "available": target_available,
        "missing": target_missing,
        "zero": zero_count,
        "negative": negative_count,
        "positive": positive_count,
        "min": safe_float(target_series.min()),
        "max": safe_float(target_series.max()),
        "mean": safe_float(target_series.mean()),
        "median": safe_float(target_series.median())
    }

    print(f"Target column       : {target}")
    print(f"Available values    : {target_available:,}")
    print(f"Missing values      : {target_missing:,}")
    print(f"Zero values         : {zero_count:,}")
    print(f"Negative values     : {negative_count:,}")
    print(f"Positive values     : {positive_count:,}")
    print(f"Minimum             : {target_series.min()}")
    print(f"Maximum             : {target_series.max()}")
    print(f"Mean                : {target_series.mean()}")
    print(f"Median              : {target_series.median()}")


# ============================================================
# PHYSICAL RANGE AUDIT
# ============================================================

print("\n" + "=" * 70)
print("9. PHYSICAL RANGE CHECK")
print("=" * 70)

range_checks = {}


def check_range(column, minimum=None, maximum=None):

    if column not in df.columns:
        return

    series = df[column].dropna()

    below_min = 0
    above_max = 0

    if minimum is not None:
        below_min = int((series < minimum).sum())

    if maximum is not None:
        above_max = int((series > maximum).sum())

    range_checks[column] = {
        "minimum_allowed": minimum,
        "maximum_allowed": maximum,
        "below_minimum": below_min,
        "above_maximum": above_max
    }

    print(
        f"{column:<30} "
        f"below_min={below_min:,}, "
        f"above_max={above_max:,}"
    )


# These are basic physical sanity checks.
# They are NOT data-cleaning rules.

check_range("poa_irradiance__5061", minimum=0)
check_range("ambient_temp__5062", minimum=-50, maximum=70)
check_range("module_temp__5063", minimum=-50, maximum=100)
check_range("RH2M", minimum=0, maximum=100)
check_range("WS10M", minimum=0)
check_range("WD10M", minimum=0, maximum=360)
check_range("CLOUD_AMT", minimum=0, maximum=100)
check_range("high_cloud_cover", minimum=0, maximum=1)
check_range("medium_cloud_cover", minimum=0, maximum=1)
check_range("low_cloud_cover", minimum=0, maximum=1)
check_range("wind_gust_10m", minimum=0)
check_range("snowfall", minimum=0)
check_range("wind_speed_900_mb", minimum=0)
check_range("wind_direction_900_mb", minimum=0, maximum=360)


# ============================================================
# SOLAR BEHAVIOUR QUICK CHECK
# ============================================================

print("\n" + "=" * 70)
print("10. SOLAR BEHAVIOUR QUICK CHECK")
print("=" * 70)

solar_behavior = {}

if (
        "poa_irradiance__5061" in df.columns
        and "ac_power__5069" in df.columns
):

    valid = df[
        [
            "poa_irradiance__5061",
            "ac_power__5069"
        ]
    ].dropna()

    night_mask = valid["poa_irradiance__5061"] <= 1
    day_mask = valid["poa_irradiance__5061"] > 1

    night_rows = int(night_mask.sum())
    day_rows = int(day_mask.sum())

    night_power_mean = safe_float(
        valid.loc[night_mask, "ac_power__5069"].mean()
    )

    day_power_mean = safe_float(
        valid.loc[day_mask, "ac_power__5069"].mean()
    )

    solar_behavior = {
        "rows_with_valid_poa_and_power": int(len(valid)),
        "low_irradiance_rows": night_rows,
        "daytime_rows": day_rows,
        "mean_power_low_irradiance": night_power_mean,
        "mean_power_daytime": day_power_mean
    }

    print(f"Valid POA + power rows : {len(valid):,}")
    print(f"Low irradiance rows    : {night_rows:,}")
    print(f"Daytime rows           : {day_rows:,}")
    print(
        f"Mean power at low POA  : "
        f"{night_power_mean}"
    )
    print(
        f"Mean daytime power     : "
        f"{day_power_mean}"
    )


# ============================================================
# ML READINESS CHECK
# ============================================================

print("\n" + "=" * 70)
print("11. INITIAL ML READINESS CHECK")
print("=" * 70)

usable_target_rows = 0

if target in df.columns:
    usable_target_rows = int(
        df[target].notna().sum()
    )

ml_ready = (
        rows > 0
        and columns == len(EXPECTED_COLUMNS)
        and duplicate_timestamp_count == 0
        and invalid_timestamps == 0
        and usable_target_rows > 0
)

print(f"Rows available        : {rows:,}")
print(f"Usable target rows    : {usable_target_rows:,}")
print(f"Duplicate timestamps  : {duplicate_timestamp_count:,}")
print(f"Invalid timestamps    : {invalid_timestamps:,}")
print(f"Initial ML readiness   : {ml_ready}")





# ============================================================
# BUILD FINAL REPORT
# ============================================================

report = {

    "dataset": {
        "file": str(INPUT_FILE),
        "rows": rows,
        "columns": columns
    },

    "schema": {
        "expected_columns": EXPECTED_COLUMNS,
        "actual_columns": df.columns.tolist(),
        "schema_exact_match": schema_match,
        "missing_columns": missing_columns,
        "unexpected_columns": unexpected_columns
    },

    "dtypes": dtype_info,

    "timestamp": {
        "start": (
            start_timestamp.isoformat()
            if start_timestamp is not None
            else None
        ),
        "end": (
            end_timestamp.isoformat()
            if end_timestamp is not None
            else None
        ),
        "invalid_timestamp_count": invalid_timestamps,
        "duplicate_timestamp_count": duplicate_timestamp_count,
        "missing_hour_count": missing_hour_count,
        "largest_gap": (
            str(largest_gap)
            if largest_gap is not None
            else None
        )
    },

    "missing_values": missing_report,

    "data_quality": {
        "reading_count_distribution":
            {
                str(k): int(v)
                for k, v in reading_count_distribution.items()
            },

        "data_quality_distribution":
            {
                str(k): int(v)
                for k, v in data_quality_distribution.items()
            }
    },

    "numeric_summary": numeric_summary,

    "target": target_report,

    "physical_range_checks": range_checks,

    "solar_behavior": solar_behavior,

    "ml_readiness": {
        "usable_target_rows": usable_target_rows,
        "initial_ml_ready": ml_ready
    }
}


# ============================================================
# SAVE REPORT
# ============================================================

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=4
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("AUDIT COMPLETED")
print("=" * 70)

print(f"\nAudit report saved at:")
print(REPORT_FILE)

print("\nNo rows were deleted.")
print("No missing values were imputed.")
print("No values were modified.")

print("\nNext step will be decided from this audit result.")