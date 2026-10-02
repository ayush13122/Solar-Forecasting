from pathlib import Path
import zipfile
import tempfile

import pandas as pd
import xarray as xr


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(".")

MAIN_CSV = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "PVDAQ_1433_NASA_Final_Hourly.csv"
)

ERA5_DIR = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "ERA5"
)

OUTPUT_CSV = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "PVDAQ_1433_NASA_ERA5_Final_Hourly.csv"
)

# System 1433 plant coordinates
PLANT_LAT = 39.7404
PLANT_LON = -105.1719


# ============================================================
# ERA5 VARIABLE MAPPING
# ============================================================

VARIABLE_MAPPING = {
    "hcc": "high_cloud_cover",
    "mcc": "medium_cloud_cover",
    "lcc": "low_cloud_cover",
    "fg10": "wind_gust_10m",
    "sf": "snowfall",
}


# ============================================================
# FUNCTION: READ ONE ERA5 ZIP FILE
# ============================================================

def read_era5_zip(zip_file):

    print(f"\nProcessing: {zip_file.name}")

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_dir = Path(temp_dir)

        # ----------------------------------------------------
        # Extract ZIP
        # ----------------------------------------------------

        with zipfile.ZipFile(zip_file, "r") as z:
            z.extractall(temp_dir)

        # ----------------------------------------------------
        # Locate inner NetCDF files
        # ----------------------------------------------------

        nc_files = list(temp_dir.glob("*.nc"))

        instant_file = None
        accum_file = None
        max_file = None

        for nc_file in nc_files:

            if "stepType-instant" in nc_file.name:
                instant_file = nc_file

            elif "stepType-accum" in nc_file.name:
                accum_file = nc_file

            elif "stepType-max" in nc_file.name:
                max_file = nc_file

        # ----------------------------------------------------
        # Read INSTANT variables
        # ----------------------------------------------------

        if instant_file is None:
            raise FileNotFoundError(
                f"Instant NetCDF file not found in {zip_file.name}"
            )

        ds_instant = xr.open_dataset(
            instant_file,
            engine="netcdf4"
        )

        # Select nearest plant grid point
        ds_instant = ds_instant.sel(
            latitude=PLANT_LAT,
            longitude=PLANT_LON,
            method="nearest"
        )

        instant_df = ds_instant[
            ["hcc", "mcc", "lcc"]
        ].to_dataframe().reset_index()

        ds_instant.close()

        # ----------------------------------------------------
        # Read ACCUMULATED snowfall
        # ----------------------------------------------------

        if accum_file is None:
            raise FileNotFoundError(
                f"Accumulated NetCDF file not found in {zip_file.name}"
            )

        ds_accum = xr.open_dataset(
            accum_file,
            engine="netcdf4"
        )

        ds_accum = ds_accum.sel(
            latitude=PLANT_LAT,
            longitude=PLANT_LON,
            method="nearest"
        )

        snowfall_df = ds_accum[
            ["sf"]
        ].to_dataframe().reset_index()

        ds_accum.close()

        # ----------------------------------------------------
        # Read MAX wind gust
        # ----------------------------------------------------

        if max_file is None:
            raise FileNotFoundError(
                f"Maximum NetCDF file not found in {zip_file.name}"
            )

        ds_max = xr.open_dataset(
            max_file,
            engine="netcdf4"
        )

        ds_max = ds_max.sel(
            latitude=PLANT_LAT,
            longitude=PLANT_LON,
            method="nearest"
        )

        gust_df = ds_max[
            ["fg10"]
        ].to_dataframe().reset_index()

        ds_max.close()

        # ----------------------------------------------------
        # Merge ERA5 components
        # ----------------------------------------------------

        df = instant_df.merge(
            snowfall_df[
                ["valid_time", "sf"]
            ],
            on="valid_time",
            how="outer"
        )

        df = df.merge(
            gust_df[
                ["valid_time", "fg10"]
            ],
            on="valid_time",
            how="outer"
        )

        # ----------------------------------------------------
        # Rename variables
        # ----------------------------------------------------

        df = df.rename(
            columns=VARIABLE_MAPPING
        )

        # ----------------------------------------------------
        # Keep only required columns
        # ----------------------------------------------------

        df = df[
            [
                "valid_time",
                "high_cloud_cover",
                "medium_cloud_cover",
                "low_cloud_cover",
                "wind_gust_10m",
                "snowfall",
            ]
        ]

        # ----------------------------------------------------
        # Remove possible duplicate timestamps
        # ----------------------------------------------------

        df = df.drop_duplicates(
            subset=["valid_time"]
        )

        print(
            f"  Extracted {len(df):,} hourly ERA5 records"
        )

        return df


# ============================================================
# MAIN PROGRAM
# ============================================================

print("=" * 70)
print("ERA5 SINGLE-LEVEL FEATURES → MASTER CSV")
print("=" * 70)


# ============================================================
# LOAD MAIN CSV
# ============================================================

print("\nLoading main PVDAQ + NASA dataset...")

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
# FIND ERA5 MONTHLY FILES
# ============================================================

era5_files = sorted(
    ERA5_DIR.glob(
        "ERA5_*_cloud_wind_snow.nc"
    )
)

print(
    f"\nERA5 monthly files found: {len(era5_files)}"
)

if len(era5_files) == 0:

    raise FileNotFoundError(
        "No ERA5 monthly files found."
    )


# ============================================================
# PROCESS ALL MONTHS
# ============================================================

all_era5 = []

for era5_file in era5_files:

    try:

        monthly_df = read_era5_zip(
            era5_file
        )

        all_era5.append(
            monthly_df
        )

    except Exception as e:

        print(
            f"  ERROR: {era5_file.name}"
        )

        print(
            f"  {e}"
        )

        raise


# ============================================================
# COMBINE ALL ERA5 DATA
# ============================================================

print("\nCombining ERA5 monthly data...")

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
# TIMESTAMP CONVERSION
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
    f"Total ERA5 hourly records: "
    f"{len(era5_df):,}"
)

print(
    f"ERA5 date range: "
    f"{era5_df['timestamp'].min()} → "
    f"{era5_df['timestamp'].max()}"
)


# ============================================================
# CHECK ERA5 MISSING VALUES
# ============================================================

print("\nERA5 missing values:")

print(
    era5_df[
        [
            "high_cloud_cover",
            "medium_cloud_cover",
            "low_cloud_cover",
            "wind_gust_10m",
            "snowfall",
        ]
    ].isna().sum()
)


# ============================================================
# CHECK DUPLICATE TIMESTAMPS
# ============================================================

duplicates = era5_df[
    "timestamp"
].duplicated().sum()

print(
    f"\nDuplicate ERA5 timestamps: {duplicates}"
)


# ============================================================
# MERGE WITH MAIN DATASET
# ============================================================

print("\nMerging ERA5 with main dataset...")

final_df = main_df.merge(
    era5_df,
    on="timestamp",
    how="left"
)


# ============================================================
# FINAL SORT
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
print("MERGE COMPLETED")
print("=" * 70)

print(
    f"\nOriginal main dataset : "
    f"{main_df.shape}"
)

print(
    f"ERA5 dataset          : "
    f"{era5_df.shape}"
)

print(
    f"Final dataset         : "
    f"{final_df.shape}"
)

print(
    f"\nOutput file:\n"
    f"{OUTPUT_CSV}"
)

print("\nNew ERA5 columns:")

for column in VARIABLE_MAPPING.values():

    print(
        f"  ✓ {column}"
    )

print("\nFinal missing values for new ERA5 columns:")

print(
    final_df[
        list(VARIABLE_MAPPING.values())
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