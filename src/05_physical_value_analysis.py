import pandas as pd

df = pd.read_csv(
    "data/processed/PVDAQ_1433_NASA_ERA5_900hPa_Final_Hourly.csv"
)

df["timestamp"] = pd.to_datetime(df["timestamp"])

# print("=" * 70)
# print("AC POWER PHYSICAL VALUE ANALYSIS")
# print("=" * 70)

negative_power = df[df["ac_power__5069"] < 0]

# print(f"\nTotal negative AC power values: {len(negative_power):,}")
#
# print("\nNegative AC power statistics:")
#
# print(
#     negative_power["ac_power__5069"].describe()
# )
#
# print("\nFirst 20 negative AC power records:")
#
# print(
#     negative_power[
#         [
#             "timestamp",
#             "ac_power__5069",
#             "poa_irradiance__5061",
#             "kwh_gross__5065",
#             "reading_count",
#             "data_quality"
#         ]
#     ]
#     .head(20)
#     .to_string(index=False)
# )

# print("\n" + "=" * 70)
# print("NEGATIVE AC POWER AT HIGH IRRADIANCE")
# print("=" * 70)
#
# high_irradiance_negative = df[
#     (df["ac_power__5069"] < 0) &
#     (df["poa_irradiance__5061"] > 100)
#     ]

# print(
#     f"\nNegative AC power with POA > 100: "
#     f"{len(high_irradiance_negative)}"
# )
#
# print(
#     high_irradiance_negative[
#         [
#             "timestamp",
#             "ac_power__5069",
#             "poa_irradiance__5061",
#             "kwh_gross__5065",
#             "reading_count",
#             "data_quality"
#         ]
#     ]
#     .to_string(index=False)
# )


# print("\n" + "=" * 70)
# print("POA IRRADIANCE PHYSICAL VALUE ANALYSIS")
# print("=" * 70)

negative_poa = df[df["poa_irradiance__5061"] < 0]

# print(f"\nTotal negative POA values: {len(negative_poa):,}")
#
# print("\nNegative POA statistics:")
#
# print(
#     negative_poa["poa_irradiance__5061"].describe()
# )
#
# print("\nFirst 20 negative POA records:")
#
# print(
#     negative_poa[
#         [
#             "timestamp",
#             "poa_irradiance__5061",
#             "ac_power__5069",
#             "kwh_gross__5065",
#             "reading_count",
#             "data_quality"
#         ]
#     ]
#     .head(20)
#     .to_string(index=False)
# )



# print("\n" + "=" * 70)
# print("POA IRRADIANCE RANGE CHECK")
# print("=" * 70)

poa = df["poa_irradiance__5061"].dropna()

# print("\nPOA statistics:")
# print(poa.describe())
#
# print("\nPOA values above 1200 W/m²:")
# print((poa > 1200).sum())
#
# print("\nPOA values above 1300 W/m²:")
# print((poa > 1300).sum())
#
# print("\nMaximum POA records:")
#
# print(
#     df[
#         ["timestamp",
#          "poa_irradiance__5061",
#          "ac_power__5069",
#          "kwh_gross__5065",
#          "reading_count",
#          "data_quality"]
#     ]
#     .sort_values("poa_irradiance__5061", ascending=False)
#     .head(10)
#     .to_string(index=False)
# )


# print("\n" + "=" * 70)
# print("AMBIENT TEMPERATURE RANGE CHECK")
# print("=" * 70)

ambient = df["ambient_temp__5062"].dropna()

# print("\nAmbient temperature statistics:")
# print(ambient.describe())
#
# print("\nMinimum temperature records:")
#
# print(
#     df[
#         ["timestamp",
#          "ambient_temp__5062",
#          "module_temp__5063",
#          "poa_irradiance__5061",
#          "ac_power__5069"]
#     ]
#     .sort_values("ambient_temp__5062")
#     .head(10)
#     .to_string(index=False)
# )

# print("\nMaximum temperature records:")
#
# print(
#     df[
#         ["timestamp",
#          "ambient_temp__5062",
#          "module_temp__5063",
#          "poa_irradiance__5061",
#          "ac_power__5069"]
#     ]
#     .sort_values("ambient_temp__5062", ascending=False)
#     .head(10)
#     .to_string(index=False)
# )


# print("\n" + "=" * 70)
# print("MODULE TEMPERATURE RANGE CHECK")
# print("=" * 70)

module = df["module_temp__5063"].dropna()

# print("\nModule temperature statistics:")
# print(module.describe())
#
# print("\nMinimum module temperature records:")
#
# print(
#     df[
#         [
#             "timestamp",
#             "module_temp__5063",
#             "ambient_temp__5062",
#             "poa_irradiance__5061",
#             "ac_power__5069"
#         ]
#     ]
#     .sort_values("module_temp__5063")
#     .head(10)
#     .to_string(index=False)
# )

# print("\nMaximum module temperature records:")
#
# print(
#     df[
#         [
#             "timestamp",
#             "module_temp__5063",
#             "ambient_temp__5062",
#             "poa_irradiance__5061",
#             "ac_power__5069"
#         ]
#     ]
#     .sort_values("module_temp__5063", ascending=False)
#     .head(10)
#     .to_string(index=False)
# )

#
# print("\n" + "=" * 70)
# print("WIND SPEED RANGE CHECK")
# print("=" * 70)

wind = df["WS10M"].dropna()

# print("\nWind speed statistics:")
# print(wind.describe())
#
# print("\nNegative wind speed values:")
# print((wind < 0).sum())
#
# print("\nVery high wind speed values (>25 m/s):")
# print((wind > 25).sum())
#
# print("\nMaximum wind speed records:")
#
# print(
#     df[
#         [
#             "timestamp",
#             "WS10M",
#             "WD10M",
#             "ac_power__5069",
#             "poa_irradiance__5061"
#         ]
#     ]
#     .sort_values("WS10M", ascending=False)
#     .head(10)
#     .to_string(index=False)
# )


# print("\n" + "=" * 70)
# print("WIND DIRECTION RANGE CHECK")
# print("=" * 70)
#
wind_direction = df["WD10M"].dropna()
#
# print("\nWind direction statistics:")
# print(wind_direction.describe())
#
# print("\nNegative wind direction values:")
# print((wind_direction < 0).sum())
#
# print("\nWind direction >= 360°:")
# print((wind_direction >= 360).sum())
#
# print("\nMinimum direction records:")
#
# print(
#     df[
#         [
#             "timestamp",
#             "WD10M",
#             "WS10M",
#             "ac_power__5069"
#         ]
#     ]
#     .sort_values("WD10M")
#     .head(10)
#     .to_string(index=False)
# )
#
# print("\nMaximum direction records:")
#
# print(
#     df[
#         [
#             "timestamp",
#             "WD10M",
#             "WS10M",
#             "ac_power__5069"
#         ]
#     ]
#     .sort_values("WD10M", ascending=False)
#     .head(10)
#     .to_string(index=False)
# )




print("\n" + "=" * 70)
print("REMAINING WEATHER / REANALYSIS FEATURE VERIFICATION")
print("=" * 70)

features = [
    "T2M",
    "RH2M",
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
    "wind_direction_900_mb"
]

for col in features:

    print("\n" + "-" * 70)
    print(f"FEATURE: {col}")
    print("-" * 70)

    print(f"Missing values: {df[col].isna().sum():,}")
    print(f"Minimum: {df[col].min()}")
    print(f"Maximum: {df[col].max()}")
    print(f"Mean: {df[col].mean()}")
    print(f"Median: {df[col].median()}")