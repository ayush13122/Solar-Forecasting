import pandas as pd

# Load the dataset
df = pd.read_csv("data/processed/PVDAQ_1433_NASA_Final_Hourly.csv")

# Total rows
# total_rows = len(df)
#
# print("Dataset shape:", df.shape)
# print("Total rows:", total_rows)
#
# # Missing value count
# missing_count = df.isnull().sum()
#
# # Missing percentage
# missing_percentage = (missing_count / total_rows) * 100
#
# # Create a summary table
# missing_summary = pd.DataFrame({
#     "Missing Values": missing_count,
#     "Missing Percentage": missing_percentage
# })
#
# print("\nMissing Value Summary:")
# print(missing_summary)






# missing = df["ac_power__5069"].isnull()
#
# print(missing.head(20))
# missing = df["ac_power__5069"].isnull()
#
# print(missing.head(10))
# print("\nAfter shift:")
# print(missing.shift().head(10))

import pandas as pd

df = pd.read_csv("data/processed/PVDAQ_1433_NASA_Final_Hourly.csv")

missing = df["ac_power__5069"].isnull()
#
change = missing.ne(missing.shift())
#
# print("Missing status:")
# print(missing.head(10))
#
# print("\nStatus change:")
# print(change.head(10))
#
# print(df.loc[change, ["timestamp", "ac_power__5069"]])
#
gap_group = change.cumsum()
#
# print(gap_group.head(20))

change.iloc[0] = False

gap_group = change.cumsum()

# print(gap_group.head(20))

# print(df.loc[640:665, [
#     "timestamp",
#     "ac_power__5069"
# ]])
#
# print("\nGap Groups:")
# print(gap_group.loc[640:665])

missing_groups = df.loc[missing, "ac_power__5069"].groupby(gap_group[missing]).size()

# print(missing_groups.head(20))
# print("Total missing gaps:", len(missing_groups))
# print("Smallest gap:", missing_groups.min(), "hours")
# print("Largest gap:", missing_groups.max(), "hours")
# print("Median gap:", missing_groups.median(), "hours")
#
# print("\nGap distribution:")
# print(missing_groups.describe())

# print("Gap size distribution:")

# print("1 hour:", (missing_groups == 1).sum())
# print("2-3 hours:", ((missing_groups >= 2) & (missing_groups <= 3)).sum())
# print("4-6 hours:", ((missing_groups >= 4) & (missing_groups <= 6)).sum())
# print("7-12 hours:", ((missing_groups >= 7) & (missing_groups <= 12)).sum())
# print("13-24 hours:", ((missing_groups >= 13) & (missing_groups <= 24)).sum())
# print("1-7 days:", ((missing_groups >= 25) & (missing_groups <= 168)).sum())
# print(">7 days:", (missing_groups > 168).sum())


# print("Gap size distribution:")
#
# print("1 hour:", (missing_groups == 1).sum())
# print("2-3 hours:", ((missing_groups >= 2) & (missing_groups <= 3)).sum())
# print("4-6 hours:", ((missing_groups >= 4) & (missing_groups <= 6)).sum())
# print("7-12 hours:", ((missing_groups >= 7) & (missing_groups <= 12)).sum())
# print("13-24 hours:", ((missing_groups >= 13) & (missing_groups <= 24)).sum())
# print("1-7 days:", ((missing_groups >= 25) & (missing_groups <= 168)).sum())
# print(">7 days:", (missing_groups > 168).sum())

# 1-hour missing gaps ke group IDs
one_hour_groups = missing_groups[missing_groups == 1]

# print("Number of 1-hour gaps:", len(one_hour_groups))
# print("\nFirst 10 one-hour gap group IDs:")
# print(one_hour_groups.head(10))

import pandas as pd

# Load final hourly dataset
df = pd.read_csv("data/processed/PVDAQ_1433_NASA_Final_Hourly.csv")

# -------------------------------------------------------
# 1. Identify missing AC power
# -------------------------------------------------------
missing = df["ac_power__5069"].isnull()

# -------------------------------------------------------
# 2. Identify where missing/non-missing status changes
# -------------------------------------------------------
change = missing.ne(missing.shift())

# First row is not an actual status change
change.iloc[0] = False

# -------------------------------------------------------
# 3. Give each continuous section a group ID
# -------------------------------------------------------
gap_group = change.cumsum()

# -------------------------------------------------------
# 4. Find groups where AC power is missing
# -------------------------------------------------------
missing_groups = (
    df.loc[missing, "ac_power__5069"]
    .groupby(gap_group[missing])
    .size()
)

# -------------------------------------------------------
# 5. Select a 1-hour missing gap
# -------------------------------------------------------
one_hour_groups = missing_groups[missing_groups == 1]

group_id = one_hour_groups.index[0]

# print("Selected gap group:", group_id)

# -------------------------------------------------------
# 6. Find start and end index of this gap
# -------------------------------------------------------
gap_indices = df.index[gap_group == group_id]

start = gap_indices.min()
end = gap_indices.max()

# -------------------------------------------------------
# 7. Show rows around the gap
# -------------------------------------------------------
features = [
    "timestamp",

    # PVDAQ plant measurements
    "ac_power__5069",
    "poa_irradiance__5061",
    "ambient_temp__5062",
    "module_temp__5063",
    "dc_voltage__5070",
    "pr__5067",
    "kwh_gross__5065",

    # NASA POWER weather
    "T2M",
    "RH2M",
    "WS10M",
    "WD10M",
    "PS",
    "PRECTOTCORR",
    "ALLSKY_SFC_SW_DWN",
    "CLOUD_AMT"
]

# Keep only columns that currently exist
features = [col for col in features if col in df.columns]

print("\nData around the missing gap:")
print(
    df.loc[
        max(0, start - 2):end + 2,
        features
    ].to_string(index=True)
)

# Previous and next available AC power
before_power = df.loc[start - 1, "ac_power__5069"]
after_power = df.loc[end + 1, "ac_power__5069"]

# Linear interpolation estimate
estimated_power = (before_power + after_power) / 2

print("Previous AC Power:", before_power)
print("Missing AC Power: NaN")
print("Next AC Power:", after_power)
print("Estimated AC Power:", estimated_power)

print("\nActual conditions during missing hour:")

print(
    df.loc[end, [
        "timestamp",
        "poa_irradiance__5061",
        "ambient_temp__5062",
        "module_temp__5063",
        "T2M",
        "RH2M",
        "WS10M",
        "PS",
        "ALLSKY_SFC_SW_DWN",
        "CLOUD_AMT"
    ]]
)