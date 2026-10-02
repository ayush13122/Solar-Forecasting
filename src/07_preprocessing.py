#Preprocessing — Step 1

# Sabse pehle hum cleaning rules ko implement karne se pehle ek controlled copy banayenge aur check karenge ki current 27-column file correctly load ho rahi hai.
import pandas as pd
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

input_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "PVDAQ_1433_NASA_ERA5_900hPa_SolarPosition_Hourly.csv"
)

df = pd.read_csv(input_path)

# print("\n" + "=" * 70)
# print("PREPROCESSING - STEP 1")
# print("=" * 70)
#
# print("\nDataset shape:")
# print(df.shape)
#
# print("\nColumns:")
# print(df.columns.tolist())
#
# print("\nTotal missing values:")
# print(df.isna().sum().sum())
#
# print("\nMissing values by column:")
# print(df.isna().sum())

# Step 2 — sirf ye chhota check--AC POWER TARGET CHECK
# Iska output basically confirm karega ki target cleaning se kitni rows affect hongi. Uske baad hum actual removal rule implement karenge.
# print("\n" + "=" * 70)
# print("STEP 2 - AC POWER TARGET CHECK")
# print("=" * 70)
#
# print("\nTotal rows:", len(df))

# print("AC Power missing:", df["ac_power__5069"].isna().sum())
#
# print(
#     "AC Power available:",
#     df["ac_power__5069"].notna().sum()
# )
#
# print(
#     "Percentage missing:",
#     round(df["ac_power__5069"].isna().mean() * 100, 2),
#     "%"
# )
#
# print(
#     "Rows remaining after removing missing target:",
#     df["ac_power__5069"].notna().sum()
# )



# Step 3: AC Power negative values cleaning.

# print("\n" + "=" * 70)
# print("STEP 3 - NEGATIVE AC POWER CHECK")
# print("=" * 70)
#
negative_ac = df[df["ac_power__5069"] < 0].copy()
#
# print("\nTotal negative AC Power values:")
# print(len(negative_ac))
#
# print("\nNegative AC Power statistics:")
# print(
#     negative_ac["ac_power__5069"].describe()
# )
#
# print("\nNegative AC Power range:")
# print(
#     "Minimum:",
#     negative_ac["ac_power__5069"].min()
# )
#
# print(
#     "Maximum:",
#     negative_ac["ac_power__5069"].max()
# )
#
# print("\nNegative AC Power with POA > 100:")
# print(
#     len(
#         negative_ac[
#             negative_ac["poa_irradiance__5061"] > 100
#             ]
#     )
# )
#
# print("\nNegative AC Power with POA <= 100:")
# print(
#     len(
#         negative_ac[
#             negative_ac["poa_irradiance__5061"] <= 100
#             ]
#     )
# )
#
# print("\nHigh-irradiance negative AC records:")
# print(
#     negative_ac[
#         negative_ac["poa_irradiance__5061"] > 100
#         ][
#         [
#             "timestamp",
#             "ac_power__5069",
#             "poa_irradiance__5061",
#             "kwh_gross__5065",
#             "reading_count",
#             "data_quality"
#         ]
#     ].to_string(index=False)
# )

# ab Step 3 ka actual cleaning action apply karte hain.
#
# Is baar sirf low-irradiance negative AC values ko 0 karenge aur high-irradiance wale 4 records ko flag karenge.

# print("\n" + "=" * 70)
# print("STEP 3 - APPLY AC POWER CLEANING")
# print("=" * 70)

# Count before cleaning
negative_before = (df["ac_power__5069"] < 0).sum()

# Low-irradiance negative AC power
low_irr_negative = (
        (df["ac_power__5069"] < 0) &
        (df["poa_irradiance__5061"] <= 100)
)

# High-irradiance negative AC power
high_irr_negative = (
        (df["ac_power__5069"] < 0) &
        (df["poa_irradiance__5061"] > 100)
)

# print("\nNegative AC before cleaning:", negative_before)
# print("Low-irradiance negatives:", low_irr_negative.sum())
# print("High-irradiance negatives:", high_irr_negative.sum())

# Convert low-irradiance negative AC to zero
df.loc[low_irr_negative, "ac_power__5069"] = 0

# Create anomaly flag
df["ac_power_negative_anomaly"] = 0
df.loc[high_irr_negative, "ac_power_negative_anomaly"] = 1

# print("\nNegative AC after cleaning:")
# print(
#     (df["ac_power__5069"] < 0).sum()
# )
#
# print("\nAnomaly flags:")
# print(
#     df["ac_power_negative_anomaly"].value_counts()
# )

# Step 4: POA Irradiance cleaning.
# print("\n" + "=" * 70)
# print("STEP 4 - NEGATIVE POA IRRADIANCE CHECK")
# print("=" * 70)
#
# negative_poa = df[df["poa_irradiance__5061"] < 0].copy()
#
# print("\nTotal negative POA values:")
# print(len(negative_poa))
#
# print("\nNegative POA statistics:")
# print(
#     negative_poa["poa_irradiance__5061"].describe()
# )
#
# print("\nNegative POA range:")
# print(
#     "Minimum:",
#     negative_poa["poa_irradiance__5061"].min()
# )
#
# print(
#     "Maximum:",
#     negative_poa["poa_irradiance__5061"].max()
# )
#
# print("\nNegative POA with AC Power > 0:")
# print(
#     len(
#         negative_poa[
#             negative_poa["ac_power__5069"] > 0
#             ]
#     )
# )
#
# print("\nNegative POA with AC Power = 0:")
# print(
#     len(
#         negative_poa[
#             negative_poa["ac_power__5069"] == 0
#             ]
#     )
# )
#
# print("\nSample negative POA records:")
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
#     ].head(20).to_string(index=False)
# )

# Hum Step 3 ko incomplete nahi chhodenge. Pehle negative AC ke saath POA relationship properly verify
# print("\n" + "=" * 70)
# print("STEP 3B - NEGATIVE AC vs POA COMPLETE CHECK")
# print("=" * 70)
#
negative_ac = df[df["ac_power__5069"] < 0]
#
# print("\nNegative AC records:", len(negative_ac))
#
# print("\nPOA missing among negative AC:")
# print(
#     negative_ac["poa_irradiance__5061"].isna().sum()
# )
#
# print("\nPOA statistics for negative AC:")
# print(
#     negative_ac["poa_irradiance__5061"].describe()
# )
#
# print("\nNegative AC grouped by POA condition:")
#
# print(
#     "POA <= 100:",
#     (
#             negative_ac["poa_irradiance__5061"] <= 100
#     ).sum()
# )

# print(
#     "POA > 100:",
#     (
#             negative_ac["poa_irradiance__5061"] > 100
#     ).sum()
# )
#
# print(
#     "POA missing:",
#     negative_ac["poa_irradiance__5061"].isna().sum()
# )

# Ab actual POA cleaning apply karte hain
# print("\n" + "=" * 70)
# print("STEP 4 - APPLY POA CLEANING RULE A")
# print("=" * 70)

# Conditions
poa_negative = df["poa_irradiance__5061"] < 0
ac_zero = df["ac_power__5069"] == 0
ac_positive = df["ac_power__5069"] > 0
ac_missing = df["ac_power__5069"].isna()

# ---------------------------------------------------------
# Rule A1: POA < 0 and AC Power == 0
# Action: POA → 0
# ---------------------------------------------------------
rule_a1 = poa_negative & ac_zero

# ---------------------------------------------------------
# Rule A2: POA < 0 and AC Power > 0
# Action: Flag anomaly, preserve original POA
# ---------------------------------------------------------
rule_a2 = poa_negative & ac_positive

# ---------------------------------------------------------
# Rule A3: POA < 0 and AC Power == NaN
# Action: Preserve original POA
# Row will be excluded later because target is missing
# ---------------------------------------------------------
rule_a3 = poa_negative & ac_missing


# Counts before cleaning
# print("\nRule A1 - Negative POA + AC = 0:")
# print(rule_a1.sum())
#
# print("\nRule A2 - Negative POA + AC > 0:")
# print(rule_a2.sum())
#
# print("\nRule A3 - Negative POA + AC = NaN:")
# print(rule_a3.sum())


# ---------------------------------------------------------
# Create anomaly flag
# ---------------------------------------------------------
df["poa_negative_anomaly"] = 0

df.loc[rule_a2, "poa_negative_anomaly"] = 1


# ---------------------------------------------------------
# Apply Rule A1
# ---------------------------------------------------------
df.loc[rule_a1, "poa_irradiance__5061"] = 0


# ---------------------------------------------------------
# Verification
# ---------------------------------------------------------
# print("\nNegative POA after cleaning:")
# print(
#     (df["poa_irradiance__5061"] < 0).sum()
# )
#
# print("\nPOA anomaly flags:")
# print(
#     df["poa_negative_anomaly"].value_counts()
# )
#
# print("\nPOA negative values with missing AC:")
# print(
#     (
#             (df["poa_irradiance__5061"] < 0) &
#             df["ac_power__5069"].isna()
#     ).sum()
# )

# Step 5: Ambient Temperature + Module Temperature + POA missing values handle

# print("\n" + "=" * 70)
# print("STEP 5 - SENSOR MISSING VALUES CHECK")
# print("=" * 70)

sensor_cols = [
    "ambient_temp__5062",
    "module_temp__5063",
    "poa_irradiance__5061"
]

# Rows where any of the three plant sensor values is missing
sensor_missing = df[sensor_cols].isna().any(axis=1)

# print("\nRows with any plant sensor missing:")
# print(sensor_missing.sum())
#
# print("\nAC Power status in sensor-missing rows:")
#
# print(
#     "AC Power available:",
#     df.loc[sensor_missing, "ac_power__5069"].notna().sum()
# )
#
# print(
#     "AC Power missing:",
#     df.loc[sensor_missing, "ac_power__5069"].isna().sum()
# )
#
# print("\nMissing values by sensor:")

# for col in sensor_cols:
#     print(
#         col,
#         "→",
#         df[col].isna().sum()
#     )

# Check whether the three sensors have exactly the same missing rows
missing_patterns = df[sensor_cols].isna().value_counts()

# print("\nMissingness patterns:")
# print(missing_patterns)

# Step 5A — sirf un 5 special rows ko inspect karte hain
# print("\n" + "=" * 70)
# print("STEP 5A - INSPECT 5 AC-AVAILABLE SENSOR-MISSING ROWS")
# print("=" * 70)
#
special_rows = sensor_missing & df["ac_power__5069"].notna()
#
# print("\nNumber of special rows:")
# print(special_rows.sum())
#
# print("\nSpecial rows:")
# print(
#     df.loc[
#         special_rows,
#         [
#             "timestamp",
#             "ac_power__5069",
#             "kwh_gross__5065",
#             "reading_count",
#             "data_quality",
#             "T2M",
#             "ALLSKY_SFC_SW_DWN",
#             "CLOUD_AMT",
#             "solar_zenith_angle",
#             "solar_azimuth_angle"
#         ]
#     ].to_string(index=False)
# )
#step 6 DC voltage check
# print("\n" + "=" * 70)
# print("STEP 6 - DC VOLTAGE FEATURE CHECK")
# print("=" * 70)
#
dc = df["dc_voltage__5070"]
#
# print("\nTotal rows:", len(df))
# print("DC Voltage missing:", dc.isna().sum())
# print("DC Voltage available:", dc.notna().sum())
# print("DC Voltage missing %:", round(dc.isna().mean() * 100, 2), "%")
#
# print("\nDC Voltage statistics:")
# print(dc.describe())
#
# print("\nDC Voltage available vs AC Power:")
# print(
#     "AC available:",
#     df.loc[dc.notna(), "ac_power__5069"].notna().sum()
# )
# print(
#     "AC missing:",
#     df.loc[dc.notna(), "ac_power__5069"].isna().sum()
# )
#
# print("\nDC Voltage available with POA > 100:")
# print(
#     (
#             dc.notna() &
#             (df["poa_irradiance__5061"] > 100)
#     ).sum()
# )
#
# print("\nDC Voltage missing with POA > 100:")
# print(
#     (
#             dc.isna() &
#             (df["poa_irradiance__5061"] > 100)
#     ).sum()
# )


# Step 6B — DC Voltage usable records
# df = pd.read_csv(input_path)
# df["timestamp"] = pd.to_datetime(df["timestamp"])
#
# print("\n" + "=" * 70)
# print("STEP 6B - DC VOLTAGE USABLE RECORDS CHECK")
# print("=" * 70)
#
# usable_dc = (
#         df["dc_voltage__5070"].notna() &
#         df["ac_power__5069"].notna() &
#         (df["poa_irradiance__5061"] > 100)
# )
#
# print("\nUsable DC records:", usable_dc.sum())
#
# print("\nYear-wise usable DC records:")
# print(
#     df.loc[usable_dc]
#     .groupby(df.loc[usable_dc, "timestamp"].dt.year)
#     .size()
# )
#
# print("\nDate range of usable DC records:")
#
# usable_dates = df.loc[usable_dc, "timestamp"]
#
# print("First:", usable_dates.min())
# print("Last :", usable_dates.max())
#dc is excluded

# Step 7 — PR check
# print("\n" + "=" * 70)
# print("STEP 7 - PR FEATURE CHECK")
# print("=" * 70)
#
# pr = df["pr__5067"]
#
# print("\nTotal rows:", len(df))
# print("PR missing:", pr.isna().sum())
# print("PR available:", pr.notna().sum())
# print("PR missing %:", round(pr.isna().mean() * 100, 2), "%")
#
# print("\nPR statistics:")
# print(pr.describe())
#
# print("\nPR extreme values:")
# print("PR > 1:", (pr > 1).sum())
# print("PR > 2:", (pr > 2).sum())
# print("PR > 10:", (pr > 10).sum())
# print("PR < 0:", (pr < 0).sum())
#
# print("\nPR available with AC Power available:")
# print(
#     (
#             pr.notna() &
#             df["ac_power__5069"].notna()
#     ).sum()
# )
#
# print("\nPR available with POA > 100:")
# print(
#     (
#             pr.notna() &
#             (df["poa_irradiance__5061"] > 100)
#     ).sum()
# )



# Ab Step 8 — remaining feature sanity check
# print("\n" + "=" * 70)
# print("STEP 8 - REMAINING FEATURE SANITY CHECK")
# print("=" * 70)
#
# check_cols = [
#     "ambient_temp__5062",
#     "module_temp__5063",
#     "T2M",
#     "RH2M",
#     "WS10M",
#     "WD10M",
#     "PS",
#     "PRECTOTCORR",
#     "ALLSKY_SFC_SW_DWN",
#     "CLOUD_AMT",
#     "high_cloud_cover",
#     "medium_cloud_cover",
#     "low_cloud_cover",
#     "wind_gust_10m",
#     "snowfall",
#     "wind_speed_900_mb",
#     "wind_direction_900_mb",
#     "solar_zenith_angle",
#     "solar_azimuth_angle"
# ]
#
# for col in check_cols:
#     print(f"\n--- {col} ---")
#     print("Missing:", df[col].isna().sum())
#     print("Min:", df[col].min())
#     print("Max:", df[col].max())
#     print("Mean:", round(df[col].mean(), 4))




# Rule 1: AC Power cleaning
# print("\n" + "=" * 70)
# print("STEP 9 - AC POWER CLEANING")
# print("=" * 70)
#
# # Identify negative AC Power values
# negative_ac = df["ac_power__5069"] < 0
#
# # Negative AC under low irradiance → treat as zero
# low_irr_negative = (
#         negative_ac &
#         (df["poa_irradiance__5061"] <= 100)
# )
#
# # Negative AC under high irradiance → preserve as anomaly
# high_irr_negative = (
#         negative_ac &
#         (df["poa_irradiance__5061"] > 100)
# )
#
# # Create anomaly flag
# df["ac_power_negative_anomaly"] = 0
#
# # Mark high-irradiance negative AC values
# df.loc[high_irr_negative, "ac_power_negative_anomaly"] = 1
#
# # Convert only low-irradiance negative AC to zero
# df.loc[low_irr_negative, "ac_power__5069"] = 0
#
# print("\nNegative AC before cleaning:", negative_ac.sum())
# print("Low-irradiance negative AC → zero:", low_irr_negative.sum())
# print("High-irradiance negative AC → anomaly:", high_irr_negative.sum())
#
# print("\nNegative AC after cleaning:",
#       (df["ac_power__5069"] < 0).sum())
#
# print("\nAnomaly flag counts:")
# print(df["ac_power_negative_anomaly"].value_counts())


# Step 10 — POA Cleaning
# print("\n" + "=" * 70)
# print("STEP 10 - POA IRRADIANCE CLEANING")
# print("=" * 70)
#
# poa_negative = df["poa_irradiance__5061"] < 0
#
# ac_zero = df["ac_power__5069"] == 0
# ac_positive = df["ac_power__5069"] > 0
# ac_missing = df["ac_power__5069"].isna()
#
# rule_a1 = poa_negative & ac_zero
# rule_a2 = poa_negative & ac_positive
# rule_a3 = poa_negative & ac_missing
#
# # Create anomaly flag
# df["poa_negative_anomaly"] = 0
#
# # Preserve negative POA when AC is positive, but flag it
# df.loc[rule_a2, "poa_negative_anomaly"] = 1
#
# # Convert negative POA to zero only when AC is zero
# df.loc[rule_a1, "poa_irradiance__5061"] = 0
#
# print("\nNegative POA before cleaning:", poa_negative.sum())
# print("Negative POA + AC = 0 → zero:", rule_a1.sum())
# print("Negative POA + AC > 0 → anomaly:", rule_a2.sum())
# print("Negative POA + AC missing → preserved:", rule_a3.sum())
#
# print("\nNegative POA after cleaning:",
#       (df["poa_irradiance__5061"] < 0).sum())
#
# print("\nPOA anomaly flag counts:")
# print(df["poa_negative_anomaly"].value_counts())

# Step 11 — Missing AC Power handling
print("\n" + "=" * 70)
print("STEP 11 - AC POWER MISSING TARGET CHECK")
print("=" * 70)

ac_missing = df["ac_power__5069"].isna()

print("\nTotal rows:", len(df))
print("AC Power missing:", ac_missing.sum())
print("AC Power available:", df["ac_power__5069"].notna().sum())
print(
    "Rows available for supervised ML:",
    df["ac_power__5069"].notna().sum()
)

print(
    "Rows excluded later due to missing target:",
    ac_missing.sum()
)

# Step 12 — Special sensor-missing rows
print("\n" + "=" * 70)
print("STEP 12 - SPECIAL SENSOR-MISSING ROWS")
print("=" * 70)

special_rows = (
        df["ac_power__5069"].notna() &
        df["ambient_temp__5062"].isna() &
        df["module_temp__5063"].isna() &
        df["poa_irradiance__5061"].isna()
)

print("\nSpecial rows:", special_rows.sum())

print("\nThese rows:")
print(
    df.loc[
        special_rows,
        [
            "timestamp",
            "ac_power__5069",
            "kwh_gross__5065",
            "reading_count",
            "data_quality",
            "T2M",
            "ALLSKY_SFC_SW_DWN"
        ]
    ].to_string(index=False)
)
# Step 13 — ML-ready dataset banane se pehle exclusion count
print("\n" + "=" * 70)
print("STEP 13 - ML EXCLUSION CHECK")
print("=" * 70)

# # Rule 1: Missing target
missing_target = df["ac_power__5069"].isna()
#
# # Rule 2: AC available but all 3 key plant sensors missing
missing_core_sensors = (
        df["ac_power__5069"].notna() &
        df["ambient_temp__5062"].isna() &
        df["module_temp__5063"].isna() &
        df["poa_irradiance__5061"].isna()
)
#
# # Combined exclusion
exclude_ml = missing_target | missing_core_sensors

print("\nMissing AC Power:", missing_target.sum())
print("AC available + all 3 core sensors missing:",
      missing_core_sensors.sum())

print("\nTotal rows excluded from ML:", exclude_ml.sum())
print("Rows remaining for ML:", (~exclude_ml).sum())

print("\nMaster dataset rows:", len(df))


# Step 14 — actual ML-ready dataset create
print("\n" + "=" * 70)
print("STEP 14 - CREATE ML-READY DATASET")
print("=" * 70)

# Create separate ML-ready dataset
ml_ready_df = df.loc[~exclude_ml].copy()

print("\nMaster dataset shape:", df.shape)
print("ML-ready dataset shape:", ml_ready_df.shape)

print("\nRows removed from ML dataset:",
      len(df) - len(ml_ready_df))

print("\nMissing AC Power in ML-ready dataset:",
      ml_ready_df["ac_power__5069"].isna().sum())

print("\nMissing core plant sensors in ML-ready dataset:")

for col in [
    "ambient_temp__5062",
    "module_temp__5063",
    "poa_irradiance__5061"
]:
    print(col, ":", ml_ready_df[col].isna().sum())


# Step 15 — ML feature selection.
# Exclude: DC Voltage, PR, timestamp, data_quality, reading_count
print("\n" + "=" * 70)
print("STEP 15 - ML FEATURE SELECTION")
print("=" * 70)

target_col = "ac_power__5069"
#
feature_cols = [
    "ambient_temp__5062",
    "module_temp__5063",
    "poa_irradiance__5061",

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

    "solar_zenith_angle",
    "solar_azimuth_angle"
]

# Create ML dataset with selected features + target
ml_df = ml_ready_df[["timestamp"] + feature_cols + [target_col]].copy()

print("\nTarget:")
print(target_col)

print("\nNumber of ML features:", len(feature_cols))

print("\nML features:")
for i, col in enumerate(feature_cols, 1):
    print(f"{i}. {col}")

print("\nML dataset shape:", ml_df.shape)

print("\nMissing values:")
print(ml_df.isna().sum().sum())





# Step 16 — Time feature creation
print("\n" + "=" * 70)
print("STEP 16 - TIME FEATURES")
print("=" * 70)
ml_df["timestamp"] = pd.to_datetime(ml_df["timestamp"])

ml_df["hour"] = ml_df["timestamp"].dt.hour
ml_df["month"] = ml_df["timestamp"].dt.month
ml_df["day_of_year"] = ml_df["timestamp"].dt.dayofyear

print("\nTime features created:")
print([
    "hour",
    "month",
    "day_of_year"
])

print("\nUpdated ML dataset shape:", ml_df.shape)

print("\nSample:")
print(
    ml_df[
        ["timestamp", "hour", "month", "day_of_year", "ac_power__5069"]
    ].head(10)
)


# Step 17 — Time feature sanity check
print("\n" + "=" * 70)
print("STEP 17 - TIME FEATURE SANITY CHECK")
print("=" * 70)

print("\nHour range:")
print(ml_df["hour"].min(), "to", ml_df["hour"].max())

print("\nHour counts:")
print(ml_df["hour"].value_counts().sort_index())

print("\nMonth range:")
print(ml_df["month"].min(), "to", ml_df["month"].max())

print("\nDay of year range:")
print(ml_df["day_of_year"].min(), "to", ml_df["day_of_year"].max())



# Step 18 — Create cyclical time features
import numpy as np

print("\n" + "=" * 70)
print("STEP 18 - CYCLICAL TIME FEATURES")
print("=" * 70)

ml_df["hour_sin"] = np.sin(2 * np.pi * ml_df["hour"] / 24)
ml_df["hour_cos"] = np.cos(2 * np.pi * ml_df["hour"] / 24)

ml_df["month_sin"] = np.sin(2 * np.pi * ml_df["month"] / 12)
ml_df["month_cos"] = np.cos(2 * np.pi * ml_df["month"] / 12)

ml_df["day_of_year_sin"] = np.sin(
    2 * np.pi * ml_df["day_of_year"] / 365.25
)
ml_df["day_of_year_cos"] = np.cos(
    2 * np.pi * ml_df["day_of_year"] / 365.25
)

print("\nCyclical features created:")
print([
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
    "day_of_year_sin", "day_of_year_cos"
])

print("\nUpdated shape:", ml_df.shape)

print("\nSample:")
print(
    ml_df[
        [
            "hour",
            "hour_sin",
            "hour_cos",
            "month",
            "month_sin",
            "month_cos"
        ]
    ].head()
)


# Step 19 — Final feature inventory check
print("\n" + "=" * 70)
print("STEP 19 - FINAL FEATURE INVENTORY CHECK")
print("=" * 70)

print("\nTotal columns:", len(ml_df.columns))

print("\nAll columns:")
for i, col in enumerate(ml_df.columns, 1):
    print(f"{i}. {col}")

print("\nMissing values:")
print("Total missing:", ml_df.isna().sum().sum())

print("\nDuplicate timestamps:")
print(ml_df["timestamp"].duplicated().sum())



# Step 20 — Leakage check

# we need to make sure no feature is effectively derived from the target or future information.
#
# Ye check model ko genuinely forecasting model banane ke liye important hai.

print("\n" + "=" * 70)
print("STEP 20 - TARGET LEAKAGE CHECK")
print("=" * 70)

target = "ac_power__5069"
#
possible_leakage_cols = [
    "kwh_gross__5065",
    "pr__5067",
    "dc_voltage__5070"
]

print("\nTarget:", target)

print("\nPotential leakage / excluded columns:")
for col in possible_leakage_cols:
    print(
        col,
        "present in ML dataframe:", col in ml_df.columns
    )

print("\nCorrelation with target:")
for col in ml_df.select_dtypes(include="number").columns:
    if col != target:
        corr = ml_df[col].corr(ml_df[target])
        print(f"{col}: {corr:.4f}")

# Step 21 — AC Power + POA outlier check
print("\n" + "=" * 70)
print("STEP 21 - TARGET & PHYSICAL OUTLIER CHECK")
print("=" * 70)

print("\nAC Power statistics:")
print(ml_df["ac_power__5069"].describe())

print("\nAC Power percentiles:")
print(
    ml_df["ac_power__5069"].quantile(
        [0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.00]
    )
)

print("\nPOA Irradiance statistics:")
print(ml_df["poa_irradiance__5061"].describe())

print("\nPOA percentiles:")
print(
    ml_df["poa_irradiance__5061"].quantile(
        [0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.00]
    )
)

print("\nAC Power > 400:")
print((ml_df["ac_power__5069"] > 400).sum())

print("\nAC Power < 0:")
print((ml_df["ac_power__5069"] < 0).sum())

print("\nPOA < 0:")
print((ml_df["poa_irradiance__5061"] < 0).sum())

print("\nPOA > 1200:")
print((ml_df["poa_irradiance__5061"] > 1200).sum())



# Step 21B karte hain. here only remaining features ka compact statistical + physical sanity check
print("\n" + "=" * 70)
print("STEP 21B - REMAINING FEATURES OUTLIER SANITY CHECK")
print("=" * 70)
#
check_cols = [
    "ambient_temp__5062",
    "module_temp__5063",
    "T2M",
    "RH2M",
    "WS10M",
    "WD10M",
    "PS",
    "PRECTOTCORR",
    "ALLSKY_SFC_SW_DWN",
    "CLOUD_AMT",
    "wind_gust_10m",
    "snowfall",
    "wind_speed_900_mb",
    "wind_direction_900_mb"
]
#
percentiles = [0.01, 0.50, 0.99]
#
for col in check_cols:

    values = ml_df[col]

    print(f"\n--- {col} ---")

    print("Min :", round(values.min(), 4))
    print("P01 :", round(values.quantile(0.01), 4))
    print("Median :", round(values.quantile(0.50), 4))
    print("P99 :", round(values.quantile(0.99), 4))
    print("Max :", round(values.max(), 4))

# Step 22 — Final feature set lock
# print("\n" + "=" * 70)
# print("STEP 22 - FINAL ML FEATURE SET")
# print("=" * 70)

final_feature_cols = [
    "ambient_temp__5062",
    "module_temp__5063",
    "poa_irradiance__5061",

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

    "solar_zenith_angle",
    "solar_azimuth_angle",

    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos"
]

target_col = "ac_power__5069"
#
final_ml_df = ml_df[
    ["timestamp"] + final_feature_cols + [target_col]
    ].copy()

# 🔒 Final outlier decision
#
# For these remaining features:
#
# No statistical outlier removal or value replacement will be performed.
#
# We'll retain the original values because the checks did not identify clearly invalid observations.

print("\nNumber of final ML features:", len(final_feature_cols))
print("Final ML dataset shape:", final_ml_df.shape)

print("\nExcluded raw time features:")
print([
    "hour",
    "month",
    "day_of_year"
])

print("\nMissing values:", final_ml_df.isna().sum().sum())


# Step 23 — Final ML dataset quality check
print("\n" + "=" * 70)
print("STEP 23 - FINAL ML DATASET QUALITY CHECK")
print("=" * 70)

print("\nShape:", final_ml_df.shape)

print("\nMissing values:")
print(final_ml_df.isna().sum().sum())

print("\nDuplicate timestamps:")
print(final_ml_df["timestamp"].duplicated().sum())

print("\nTimestamp range:")
print("First:", final_ml_df["timestamp"].min())
print("Last :", final_ml_df["timestamp"].max())

print("\nTarget statistics:")
print(final_ml_df["ac_power__5069"].describe())

print("\nFinal feature count:", len(final_feature_cols))

print("\nFinal features:")
for i, col in enumerate(final_feature_cols, 1):
    print(f"{i}. {col}")

# Step 24 — Final negative-value lock
print("\n" + "=" * 70)
print("STEP 24 - FINAL NEGATIVE VALUE LOCK")
print("=" * 70)

# Count before cleaning
negative_ac_before = (
        final_ml_df["ac_power__5069"] < 0
).sum()

negative_poa_before = (
        final_ml_df["poa_irradiance__5061"] < 0
).sum()

print("\nNegative AC before:", negative_ac_before)
print("Negative POA before:", negative_poa_before)

# Final lock: negative generation/irradiance treated as zero
final_ml_df.loc[
    final_ml_df["ac_power__5069"] < 0,
    "ac_power__5069"
] = 0

final_ml_df.loc[
    final_ml_df["poa_irradiance__5061"] < 0,
    "poa_irradiance__5061"
] = 0

# # Verify
print("\nNegative AC after:",
      (final_ml_df["ac_power__5069"] < 0).sum())

print("Negative POA after:",
      (final_ml_df["poa_irradiance__5061"] < 0).sum())

# 🚀 Step 25 — Final CSV save + verification
print("\n" + "=" * 70)
print("STEP 25 - SAVE & VERIFY FINAL PREPROCESSED DATASET")
print("=" * 70)

output_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "PVDAQ_1433_ML_Ready_Preprocessed_Hourly.csv"
)
#
# Save
# final_ml_df.to_csv(output_path, index=False)
#
# print("\nFile saved successfully!")
# print("Path:", output_path)
# print("Saved shape:", final_ml_df.shape)
#
# # Reload saved file
check_df = pd.read_csv(output_path)

print("\n--- RELOAD VERIFICATION ---")

print("Reloaded shape:", check_df.shape)

print(
    "Missing values:",
    check_df.isna().sum().sum()
)

print(
    "Duplicate timestamps:",
    check_df["timestamp"].duplicated().sum()
)

print(
    "Negative AC Power:",
    (check_df["ac_power__5069"] < 0).sum()
)

print(
    "Negative POA:",
    (check_df["poa_irradiance__5061"] < 0).sum()
)

# Step 26 start karte hain — 1-hour-ahead forecasting dataset.
print("\n" + "=" * 70)
print("STEP 26 - CREATE 1-HOUR-AHEAD FORECASTING DATASET")
print("=" * 70)

forecast_df = final_ml_df.copy()

# Next-hour timestamp and target
forecast_df["target_timestamp"] = forecast_df["timestamp"].shift(-1)
forecast_df["target_ac_power"] = forecast_df["ac_power__5069"].shift(-1)

# Check whether next available row is exactly 1 hour later
time_gap = (
        forecast_df["target_timestamp"] - forecast_df["timestamp"]
)

valid_1h = time_gap == pd.Timedelta(hours=1)

print("\nTotal candidate rows:", len(forecast_df))
print("Exact 1-hour pairs:", valid_1h.sum())
print("Non-1-hour pairs:", (~valid_1h).sum())

# Keep only genuine t -> t+1 hour pairs
forecast_df = forecast_df.loc[valid_1h].copy()

print("\nForecasting dataset shape:", forecast_df.shape)

print("\nFirst 5 forecasting pairs:")
print(
    forecast_df[
        ["timestamp", "ac_power__5069",
         "target_timestamp", "target_ac_power"]
    ].head()
)

print("\nLast 5 forecasting pairs:")
print(
    forecast_df[
        ["timestamp", "ac_power__5069",
         "target_timestamp", "target_ac_power"]
    ].tail()
)


# Step 27 karte hain — X/y finalization.
print("\n" + "=" * 70)
print("STEP 27 - FINALIZE FORECASTING FEATURES (X) AND TARGET (y)")
print("=" * 70)

# Current-hour AC Power is allowed because we are predicting NEXT hour
forecast_feature_cols = [
    "ac_power__5069",
    "ambient_temp__5062",
    "module_temp__5063",
    "poa_irradiance__5061",
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
    "solar_zenith_angle",
    "solar_azimuth_angle",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos"
]

target_col_forecast = "target_ac_power"

X = forecast_df[forecast_feature_cols].copy()
y = forecast_df[target_col_forecast].copy()

print("\nX shape:", X.shape)
print("y shape:", y.shape)

print("\nNumber of features:", len(forecast_feature_cols))

print("\nMissing values in X:", X.isna().sum().sum())
print("Missing values in y:", y.isna().sum())

print("\nTarget statistics:")
print(y.describe())

print("\nFirst 5 X rows:")
print(X.head())

print("\nFirst 5 y values:")
print(y.head())