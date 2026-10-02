import pandas as pd
import pvlib

# Load final master dataset
df = pd.read_csv(
    "data/processed/PVDAQ_1433_NASA_ERA5_900hPa_Final_Hourly.csv"
)

df["timestamp"] = pd.to_datetime(df["timestamp"])

# PVDAQ System 1433 location
LATITUDE = 39.7404
LONGITUDE = -105.1719

# Solar position calculation
solar_position = pvlib.solarposition.get_solarposition(
    time=df["timestamp"],
    latitude=LATITUDE,
    longitude=LONGITUDE
)

# Add calculated features
df["solar_zenith_angle"] = solar_position["zenith"].values
df["solar_azimuth_angle"] = solar_position["azimuth"].values

# Save as a new file
output_path = (
    "data/processed/"
    "PVDAQ_1433_NASA_ERA5_900hPa_SolarPosition_Hourly.csv"
)

df.to_csv(output_path, index=False)

# print("=" * 70)
# print("SOLAR POSITION CALCULATION")
# print("=" * 70)
#
# print(f"\nRows: {len(df):,}")
#
# print("\nNew columns:")
# print("solar_zenith_angle")
# print("solar_azimuth_angle")
#
# print("\nFirst 10 calculated values:")
#
# print(
#     df[
#         [
#             "timestamp",
#             "solar_zenith_angle",
#             "solar_azimuth_angle"
#         ]
#     ]
#     .head(10)
#     .to_string(index=False)
# )

# print(f"\nSaved to:")
# print(output_path)
#
# print("\nTimestamp information:")
# print(df["timestamp"].dtype)
# print(df["timestamp"].head())
#
# print("\nSolar position timezone check:")
# print(solar_position.head())

# print("\n" + "=" * 70)
# print("TIMESTAMP CONTINUITY / DST CHECK")
# print("=" * 70)
#
df = df.sort_values("timestamp")

df["time_diff"] = df["timestamp"].diff()
#
# print("\nTime difference distribution:")
# print(df["time_diff"].value_counts().head(10))
#
# print("\nRows around March 2011:")
# print(
#     df[
#         (df["timestamp"] >= "2011-03-12") &
#         (df["timestamp"] <= "2011-03-14")
#         ][["timestamp"]]
#     .to_string(index=False)
# )
#
# print("\nRows around November 2011:")
# print(
#     df[
#         (df["timestamp"] >= "2011-11-05") &
#         (df["timestamp"] <= "2011-11-07")
#         ][["timestamp"]]
#     .to_string(index=False)
# )



# print("\n" + "=" * 70)
# print("TIMESTAMP vs SOLAR GENERATION CHECK")
# print("=" * 70)
#
sample = df[
    (df["timestamp"] >= "2011-06-20") &
    (df["timestamp"] < "2011-06-21")
    ][
    [
        "timestamp",
        "poa_irradiance__5061",
        "ac_power__5069"
    ]
]
#
# print(sample.to_string(index=False))



import pandas as pd
import pvlib

# Load final master dataset
df = pd.read_csv(
    "data/processed/PVDAQ_1433_NASA_ERA5_900hPa_Final_Hourly.csv"
)

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# PVDAQ System 1433 location
LATITUDE = 39.7404
LONGITUDE = -105.1719

# ---------------------------------------------------------
# Convert PVDAQ local timestamps to MST (UTC-7)
# ---------------------------------------------------------
mst_time = df["timestamp"].dt.tz_localize("Etc/GMT+7")

# ---------------------------------------------------------
# Calculate solar position
# ---------------------------------------------------------
solar_position = pvlib.solarposition.get_solarposition(
    time=mst_time,
    latitude=LATITUDE,
    longitude=LONGITUDE
)

# Add solar position features
df["solar_zenith_angle"] = solar_position["zenith"].values
df["solar_azimuth_angle"] = solar_position["azimuth"].values

# ---------------------------------------------------------
# Basic verification
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("SOLAR POSITION VERIFICATION")
print("=" * 70)

print("\nDataset shape:")
print(df.shape)

print("\nSolar position missing values:")
print(
    df[
        ["solar_zenith_angle", "solar_azimuth_angle"]
    ].isna().sum()
)

print("\nSolar Zenith statistics:")
print(df["solar_zenith_angle"].describe())

print("\nSolar Azimuth statistics:")
print(df["solar_azimuth_angle"].describe())

print("\nFirst 10 calculated values:")
print(
    df[
        [
            "timestamp",
            "solar_zenith_angle",
            "solar_azimuth_angle"
        ]
    ].head(10).to_string(index=False)
)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
output_path = (
    "data/processed/"
    "PVDAQ_1433_NASA_ERA5_900hPa_SolarPosition_Hourly.csv"
)

df.to_csv(output_path, index=False)

print("\nSaved successfully:")
print(output_path)