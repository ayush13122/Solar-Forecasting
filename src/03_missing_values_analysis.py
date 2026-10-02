import pandas as pd

df = pd.read_csv(
    "data/processed/PVDAQ_1433_NASA_ERA5_900hPa_Final_Hourly.csv"
)
df["timestamp"] = pd.to_datetime(df["timestamp"])
print("=" * 70)
print("MISSING VALUES ANALYSIS")
print("=" * 70)

missing_count = df.isnull().sum()
missing_percent = (missing_count / len(df)) * 100

missing_summary = pd.DataFrame({
    "Missing Count": missing_count,
    "Missing Percentage": missing_percent.round(2)
})

print("\n", missing_summary)

print("\n" + "=" * 70)
print("AC POWER MISSING VALUE PATTERN")
print("=" * 70)

missing_power = df[df["ac_power__5069"].isna()]

print(f"\nTotal missing AC Power: {len(missing_power):,}")

print("\nFirst 20 missing AC Power timestamps:")
print(
    missing_power[
        ["timestamp", "poa_irradiance__5061", "kwh_gross__5065", "reading_count", "data_quality"]
    ]
    .head(20)
    .to_string(index=False)
)


missing_power = df[df["ac_power__5069"].isna()].copy()

missing_power["time_diff"] = (
    missing_power["timestamp"].diff()
)

print("\n" + "=" * 70)
print("AC POWER MISSING GAPS")
print("=" * 70)

print("\nTime difference between consecutive missing records:")
print(
    missing_power["time_diff"]
    .value_counts()
    .head(15)
)


missing_power = df[df["ac_power__5069"].isna()].copy()

missing_power["gap"] = missing_power["timestamp"].diff()

missing_power["group"] = (
    (missing_power["gap"] != pd.Timedelta(hours=1))
    .cumsum()
)

gap_summary = (
    missing_power
    .groupby("group")
    .agg(
        start_time=("timestamp", "min"),
        end_time=("timestamp", "max"),
        missing_hours=("timestamp", "count")
    )
    .reset_index(drop=True)
)
#
print("\n" + "=" * 70)
print("CONTINUOUS AC POWER MISSING BLOCKS")
print("=" * 70)

print(f"\nTotal missing blocks: {len(gap_summary):,}")

print("\nFirst 20 missing blocks:")
print(
    gap_summary
    .head(20)
    .to_string(index=False)
)

print("\nLargest missing blocks:")
print(
    gap_summary
    .sort_values("missing_hours", ascending=False)
    .head(10)
    .to_string(index=False)
)
#
#
#

print("\n" + "=" * 70)
print("AC POWER MISSING GAP DURATION")
print("=" * 70)

def classify_gap(hours):
    if hours == 1:
        return "1 hour"
    elif hours <= 5:
        return "2-5 hours"
    elif hours <= 24:
        return "6-24 hours"
    else:
        return ">24 hours"

gap_summary["gap_category"] = gap_summary["missing_hours"].apply(classify_gap)

print(
    gap_summary["gap_category"]
    .value_counts()
    .reindex(["1 hour", "2-5 hours", "6-24 hours", ">24 hours"])
)

print("\n" + "=" * 70)
print("WEATHER / SENSOR MISSING VALUE PATTERN")
print("=" * 70)

columns = [
    "ambient_temp__5062",
    "module_temp__5063",
    "poa_irradiance__5061"
]

for col in columns:
    missing = df[df[col].isna()]

    print(f"\n{col}")
    print(f"Missing values: {len(missing):,}")

    print("\nFirst 10 missing records:")
    print(
        missing[
            [
                "timestamp",
                "ac_power__5069",
                "ambient_temp__5062",
                "module_temp__5063",
                "poa_irradiance__5061",
                "reading_count",
                "data_quality"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )



    print("\n" + "=" * 70)
print("MISSING TIMESTAMP OVERLAP")
print("=" * 70)

for col in columns:
    print(
        col,
        "missing timestamps:",
        df.loc[df[col].isna(), "timestamp"].nunique()
    )

same_missing = (
        df["ambient_temp__5062"].isna()
        & df["module_temp__5063"].isna()
        & df["poa_irradiance__5061"].isna()
)

print("\nAll three missing together:", same_missing.sum())


print("\n" + "=" * 70)
print("EXTERNAL WEATHER DATA DURING SENSOR MISSING PERIODS")
print("=" * 70)

sensor_missing = (
        df["ambient_temp__5062"].isna()
        & df["module_temp__5063"].isna()
        & df["poa_irradiance__5061"].isna()
)

weather_columns = [
    "T2M",
    "RH2M",
    "WS10M",
    "ALLSKY_SFC_SW_DWN",
    "CLOUD_AMT",
    "high_cloud_cover",
    "medium_cloud_cover",
    "low_cloud_cover"
]

print(f"\nSensor-missing hours: {sensor_missing.sum():,}")

print("\nMissing values in external weather data during these hours:")

print(
    df.loc[sensor_missing, weather_columns]
    .isna()
    .sum()
)

print("\n" + "=" * 70)
print("SENSOR MISSING GAP DURATION")
print("=" * 70)

sensor_missing = df[
    df["poa_irradiance__5061"].isna()
].copy()

sensor_missing["gap"] = sensor_missing["timestamp"].diff()

sensor_missing["group"] = (
    (sensor_missing["gap"] != pd.Timedelta(hours=1))
    .cumsum()
)

sensor_gap_summary = (
    sensor_missing
    .groupby("group")
    .agg(
        start_time=("timestamp", "min"),
        end_time=("timestamp", "max"),
        missing_hours=("timestamp", "count")
    )
    .reset_index(drop=True)
)

print(f"\nTotal sensor missing blocks: {len(sensor_gap_summary):,}")

print("\nLargest sensor missing blocks:")
print(
    sensor_gap_summary
    .sort_values("missing_hours", ascending=False)
    .head(10)
    .to_string(index=False)
)

print("\n" + "=" * 70)
print("SENSOR MISSING GAP DURATION")
print("=" * 70)

def classify_sensor_gap(hours):
    if hours == 1:
        return "1 hour"
    elif hours <= 5:
        return "2-5 hours"
    elif hours <= 24:
        return "6-24 hours"
    else:
        return ">24 hours"

sensor_gap_summary["gap_category"] = (
    sensor_gap_summary["missing_hours"]
    .apply(classify_sensor_gap)
)

print(
    sensor_gap_summary["gap_category"]
    .value_counts()
    .reindex(["1 hour", "2-5 hours", "6-24 hours", ">24 hours"])
)


print("\n" + "=" * 70)
print("SHORT SENSOR GAP - BEFORE / AFTER VALUES")
print("=" * 70)

short_gaps = sensor_gap_summary[
    sensor_gap_summary["missing_hours"] <= 24
    ]

for _, gap in short_gaps.head(10).iterrows():

    start = gap["start_time"]
    end = gap["end_time"]

    before = df[df["timestamp"] < start].tail(1)
    after = df[df["timestamp"] > end].head(1)

    print("\n" + "-" * 70)
    print(f"Gap: {start} → {end}")
    print(f"Missing hours: {gap['missing_hours']}")

    print("\nBefore:")
    print(
        before[
            [
                "timestamp",
                "ambient_temp__5062",
                "module_temp__5063",
                "poa_irradiance__5061"
            ]
        ].to_string(index=False)
    )

    print("\nAfter:")
    print(
        after[
            [
                "timestamp",
                "ambient_temp__5062",
                "module_temp__5063",
                "poa_irradiance__5061"
            ]
        ].to_string(index=False)
    )
    print("\n" + "=" * 70)
print("SHORT SENSOR GAPS: DAY vs NIGHT")
print("=" * 70)




short_gap_hours = []

for _, gap in short_gaps.iterrows():

    start = gap["start_time"]
    end = gap["end_time"]

    gap_data = df[
        (df["timestamp"] >= start) &
        (df["timestamp"] <= end)
        ]

    # Missing block ke just before aur after irradiance
    before = df[df["timestamp"] < start].tail(1)
    after = df[df["timestamp"] > end].head(1)

    before_poa = before["poa_irradiance__5061"].iloc[0]
    after_poa = after["poa_irradiance__5061"].iloc[0]

    if before_poa <= 1 and after_poa <= 1:
        condition = "Night-like"
    else:
        condition = "Day / Transition"

    short_gap_hours.append(condition)

print("\nShort-gap classification:")
print(
    pd.Series(short_gap_hours).value_counts()
)



print("=" * 70)
print("DC VOLTAGE MISSING VALUE ANALYSIS")
print("=" * 70)

missing_dc = df[df["dc_voltage__5070"].isna()]

print(f"\nTotal missing DC Voltage: {len(missing_dc):,}")

print("\nFirst 20 missing DC Voltage records:")

print(
    missing_dc[
        [
            "timestamp",
            "dc_voltage__5070",
            "ac_power__5069",
            "poa_irradiance__5061",
            "reading_count",
            "data_quality"
        ]
    ]
    .head(20)
    .to_string(index=False)
)


missing_dc["gap"] = missing_dc["timestamp"].diff()

missing_dc["group"] = (
    (missing_dc["gap"] != pd.Timedelta(hours=1))
    .cumsum()
)

dc_gap_summary = (
    missing_dc
    .groupby("group")
    .agg(
        start_time=("timestamp", "min"),
        end_time=("timestamp", "max"),
        missing_hours=("timestamp", "count")
    )
    .reset_index(drop=True)
)

print("\n" + "=" * 70)
print("DC VOLTAGE MISSING BLOCKS")
print("=" * 70)

print(f"\nTotal missing blocks: {len(dc_gap_summary):,}")

print("\nLargest missing blocks:")
print(
    dc_gap_summary
    .sort_values("missing_hours", ascending=False)
    .head(20)
    .to_string(index=False)
)





print("\n" + "=" * 70)
print("DC VOLTAGE AVAILABLE VALUES")
print("=" * 70)

dc_available = df[df["dc_voltage__5070"].notna()]

print(f"\nAvailable DC Voltage values: {len(dc_available):,}")

print("\nDC Voltage statistics:")
print(
    dc_available["dc_voltage__5070"].describe()
)

print("\nFirst 20 available DC Voltage records:")
print(
    dc_available[
        [
            "timestamp",
            "dc_voltage__5070",
            "ac_power__5069",
            "poa_irradiance__5061"
        ]
    ]
    .head(20)
    .to_string(index=False)
)


print("\n" + "=" * 70)
print("DC VOLTAGE: DAY vs NIGHT")
print("=" * 70)

dc_available["condition"] = dc_available[
    "poa_irradiance__5061"
].apply(
    lambda x: "Night / Near-zero irradiance"
    if x <= 1
    else "Day / Irradiance present"
)

print(
    dc_available["condition"].value_counts()
)

print("\n" + "=" * 70)
print("DC VOLTAGE vs AC POWER AVAILABILITY")
print("=" * 70)

dc_available = df[df["dc_voltage__5070"].notna()]

print("\nWhen DC Voltage is available:")

print(
    "AC Power available:",
    dc_available["ac_power__5069"].notna().sum()
)

print(
    "AC Power missing:",
    dc_available["ac_power__5069"].isna().sum()
)

print("\nAC Power availability percentage:")

print(
    (dc_available["ac_power__5069"].notna().mean() * 100).round(2),
    "%"
)






print("\n" + "=" * 70)
print("PR MISSING VALUE ANALYSIS")
print("=" * 70)

missing_pr = df[df["pr__5067"].isna()]

print(f"\nTotal missing PR: {len(missing_pr):,}")

print("\nFirst 20 missing PR records:")

print(
    missing_pr[
        [
            "timestamp",
            "pr__5067",
            "ac_power__5069",
            "poa_irradiance__5061",
            "kwh_gross__5065",
            "reading_count",
            "data_quality"
        ]
    ]
    .head(20)
    .to_string(index=False)
)

print("\n" + "=" * 70)
print("PR MISSING BLOCK ANALYSIS")
print("=" * 70)

missing_pr = df[df["pr__5067"].isna()].copy()

missing_pr["time_diff"] = missing_pr["timestamp"].diff()

missing_pr["new_block"] = (
    (missing_pr["time_diff"] != pd.Timedelta(hours=1))
    .astype(int)
)

missing_pr["block_id"] = missing_pr["new_block"].cumsum()

blocks = (
    missing_pr
    .groupby("block_id")
    .agg(
        start=("timestamp", "min"),
        end=("timestamp", "max"),
        hours=("timestamp", "count")
    )
    .sort_values("hours", ascending=False)
)

print(f"\nTotal PR missing blocks: {len(blocks):,}")

print("\nTop 15 largest PR missing blocks:")

print(
    blocks
    .head(15)
    .to_string(index=False)
)

print("\nPR missing block duration distribution:")

print(
    pd.cut(
        blocks["hours"],
        bins=[0, 1, 5, 24, float("inf")],
        labels=["1 hour", "2-5 hours", "6-24 hours", ">24 hours"]
    )
    .value_counts()
    .sort_index()
)


print("\n" + "=" * 70)
print("PR AVAILABLE VALUE ANALYSIS")
print("=" * 70)

pr_available = df[df["pr__5067"].notna()]["pr__5067"]

print(f"\nAvailable PR values: {len(pr_available):,}")

print("\nPR statistics:")
print(pr_available.describe())

print("\nPR values greater than 1:")
print((pr_available > 1).sum())

print("\nPR values greater than 2:")
print((pr_available > 2).sum())

print("\nPR values greater than 10:")
print((pr_available > 10).sum())

print("\nPR values less than 0:")
print((pr_available < 0).sum())

print("\n" + "=" * 70)
print("EXTREME PR VALUE ANALYSIS")
print("=" * 70)

extreme_pr = df[df["pr__5067"] > 10]

print(f"\nPR values > 10: {len(extreme_pr)}")

print("\nExtreme PR records:")

print(
    extreme_pr[
        [
            "timestamp",
            "pr__5067",
            "ac_power__5069",
            "poa_irradiance__5061",
            "kwh_gross__5065",
            "dc_voltage__5070",
            "reading_count",
            "data_quality"
        ]
    ]
    .sort_values("pr__5067", ascending=False)
    .to_string(index=False)
)

print("\n" + "=" * 70)
print("PR EXTREME VALUES BY YEAR")
print("=" * 70)

extreme_pr = df[df["pr__5067"] > 10].copy()

extreme_pr["year"] = extreme_pr["timestamp"].dt.year

print(
    extreme_pr["year"]
    .value_counts()
    .sort_index()
)