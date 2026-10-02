from pathlib import Path
import zipfile
import tempfile

import pandas as pd
import xarray as xr
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(".")

MAIN_CSV = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "PVDAQ_1433_NASA_ERA5_Final_Hourly.csv"
)

ERA5_DIR = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "ERA5_900hPa"
)

OUTPUT_CSV = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "PVDAQ_1433_NASA_ERA5_900hPa_Final_Hourly.csv"
)

# System 1433 coordinates
PLANT_LAT = 39.7404
PLANT_LON = -105.1719


# ============================================================
# FUNCTION: READ ONE 900 hPa FILE
# ============================================================

def read_900hpa_file(nc_file):

    print(f"\nProcessing: {nc_file.name}")

    ds = xr.open_dataset(
        nc_file,
        engine="netcdf4"
    )

    # --------------------------------------------------------
    # Select nearest ERA5 grid point
    # --------------------------------------------------------

    ds = ds.sel(
        latitude=PLANT_LAT,
        longitude=PLANT_LON,
        method="nearest"
    )

    # --------------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------------

    df = ds[
        ["u", "v"]
    ].to_dataframe().reset_index()

    ds.close()

    # --------------------------------------------------------
    # Calculate wind speed
    #
    # Speed = sqrt(U² + V²)
    # --------------------------------------------------------

    df["wind_speed_900_mb"] = np.sqrt(
        df["u"] ** 2 +
        df["v"] ** 2
    )

    # --------------------------------------------------------
    # Calculate wind direction
    #
    # Meteorological direction:
    # direction wind is coming FROM
    # --------------------------------------------------------

    df["wind_direction_900_mb"] = (
            np.degrees(
                np.arctan2(
                    -df["u"],
                    -df["v"]
                )
            ) % 360
    )

    # --------------------------------------------------------
    # Keep required columns only
    # --------------------------------------------------------

    df = df[
        [
            "valid_time",
            "wind_speed_900_mb",
            "wind_direction_900_mb"
        ]
    ]

    return df


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("ERA5 900 hPa WIND → MASTER CSV")
print("=" * 70)


# ============================================================
# LOAD MAIN DATASET
# ============================================================

print("\nLoading existing PVDAQ + NASA + ERA5 dataset...")

main_df = pd.read_csv(
    MAIN_CSV,
    parse_dates=["timestamp"]
)

print(
    f"Main dataset shape: {main_df.shape}"
)

print(
    f"Main date range: "
    f"{main_df['timestamp'].min()} → "
    f"{main_df['timestamp'].max()}"
)


# ============================================================
# FIND 900 hPa FILES
# ============================================================

era5_files = sorted(
    ERA5_DIR.glob(
        "ERA5_900hPa_*.nc"
    )
)

print(
    f"\n900 hPa monthly files found: "
    f"{len(era5_files)}"
)

if len(era5_files) == 0:

    raise FileNotFoundError(
        "No 900 hPa ERA5 files found."
    )


# ============================================================
# PROCESS ALL MONTHS
# ============================================================

all_era5 = []

for era5_file in era5_files:

    monthly_df = read_900hpa_file(
        era5_file
    )

    all_era5.append(
        monthly_df
    )

    print(
        f"  Extracted "
        f"{len(monthly_df):,} hourly records"
    )


# ============================================================
# COMBINE
# ============================================================

print("\nCombining 900 hPa monthly data...")

era5_df = pd.concat(
    all_era5,
    ignore_index=True
)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

era5_df = era5_df.drop_duplicates(
    subset=["valid_time"]
)


# ============================================================
# TIMESTAMP
# ============================================================

era5_df["timestamp"] = pd.to_datetime(
    era5_df["valid_time"],
    utc=True
).dt.tz_localize(None)

era5_df = era5_df.drop(
    columns=["valid_time"]
)


# ============================================================
# SORT
# ============================================================

era5_df = era5_df.sort_values(
    "timestamp"
).reset_index(drop=True)


print(
    f"\nTotal 900 hPa records: "
    f"{len(era5_df):,}"
)

print(
    f"900 hPa date range: "
    f"{era5_df['timestamp'].min()} → "
    f"{era5_df['timestamp'].max()}"
)


# ============================================================
# CHECK MISSING VALUES
# ============================================================

print("\n900 hPa missing values:")

print(
    era5_df[
        [
            "wind_speed_900_mb",
            "wind_direction_900_mb"
        ]
    ].isna().sum()
)


# ============================================================
# CHECK DUPLICATES
# ============================================================

print(
    "\nDuplicate 900 hPa timestamps:",
    era5_df["timestamp"].duplicated().sum()
)


# ============================================================
# MERGE
# ============================================================

print("\nMerging 900 hPa features...")

final_df = main_df.merge(
    era5_df,
    on="timestamp",
    how="left"
)


# ============================================================
# SORT FINAL DATASET
# ============================================================

final_df = final_df.sort_values(
    "timestamp"
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)

final_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("900 hPa MERGE COMPLETED")
print("=" * 70)

print(
    f"\nOriginal dataset : {main_df.shape}"
)

print(
    f"900 hPa dataset  : {era5_df.shape}"
)

print(
    f"Final dataset    : {final_df.shape}"
)

print(
    f"\nOutput file:\n{OUTPUT_CSV}"
)

print("\nNew columns:")

print("  ✓ wind_speed_900_mb")
print("  ✓ wind_direction_900_mb")

print("\nFinal missing values:")

print(
    final_df[
        [
            "wind_speed_900_mb",
            "wind_direction_900_mb"
        ]
    ].isna().sum()
)

print("\nFinal date range:")

print(
    final_df["timestamp"].min(),
    "→",
    final_df["timestamp"].max()
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)