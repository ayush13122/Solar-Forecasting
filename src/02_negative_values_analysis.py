import pandas as pd

# Load final master dataset
df = pd.read_csv(
    "data/processed/PVDAQ_1433_NASA_ERA5_900hPa_Final_Hourly.csv"
)

# Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])


# ============================================================
# 1. NEGATIVE POA IRRADIANCE
# ============================================================

negative_poa = df[df["poa_irradiance__5061"] < 0]

print("=" * 70)
print("NEGATIVE POA IRRADIANCE ANALYSIS")
print("=" * 70)

print(f"\nTotal negative POA values: {len(negative_poa):,}")

print("\nFirst 20 negative POA records:")
print(
    negative_poa[
        [
            "timestamp",
            "poa_irradiance__5061",
            "ac_power__5069",
            "ambient_temp__5062",
            "module_temp__5063"
        ]
    ].head(20).to_string(index=False)
)


# ============================================================
# 2. NEGATIVE AC POWER
# ============================================================

negative_power = df[df["ac_power__5069"] < 0]

print("\n" + "=" * 70)
print("NEGATIVE AC POWER ANALYSIS")
print("=" * 70)

print(f"\nTotal negative AC power values: {len(negative_power):,}")

print("\nFirst 20 negative AC power records:")
print(
    negative_power[
        [
            "timestamp",
            "ac_power__5069",
            "poa_irradiance__5061",
            "ambient_temp__5062",
            "module_temp__5063"
        ]
    ].head(20).to_string(index=False)
)

# # ============================================================
# # 3. NEGATIVE POA RANGE
# # ============================================================
#
print("\n" + "=" * 70)
print("NEGATIVE POA RANGE")
print("=" * 70)

print("\nNegative POA statistics:")

print(
    negative_poa["poa_irradiance__5061"].describe()
)


# ============================================================
# 4. NEGATIVE POWER RANGE
# ============================================================

print("\n" + "=" * 70)
print("NEGATIVE AC POWER RANGE")
print("=" * 70)

print("\nNegative AC power statistics:")

print(
    negative_power["ac_power__5069"].describe()
)


# ============================================================
# 5. NEGATIVE POWER + IRRADIANCE
# ============================================================

print("\n" + "=" * 70)
print("NEGATIVE POWER WITH IRRADIANCE")
print("=" * 70)

print("\nPOA statistics for negative power records:")

print(
    negative_power["poa_irradiance__5061"].describe()
)





# ============================================================
# 6. NEGATIVE POWER DURING HIGH IRRADIANCE
# ============================================================

print("\n" + "=" * 70)
print("NEGATIVE POWER DURING HIGH IRRADIANCE")
print("=" * 70)

high_irradiance_negative_power = negative_power[
    negative_power["poa_irradiance__5061"] > 100
    ]

print(
    f"\nNegative power records with POA > 100: "
    f"{len(high_irradiance_negative_power):,}"
)

print("\nThese records:")

print(
    high_irradiance_negative_power[
        [
            "timestamp",
            "ac_power__5069",
            "poa_irradiance__5061",
            "ambient_temp__5062",
            "module_temp__5063"
        ]
    ]
    .sort_values("poa_irradiance__5061", ascending=False)
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 7. DETAILED INVESTIGATION OF HIGH-IRRADIANCE
#    NEGATIVE POWER RECORDS
# ============================================================

print("\n" + "=" * 70)
print("DETAILED INVESTIGATION")
print("=" * 70)

columns_to_check = [
    "timestamp",
    "ac_power__5069",
    "poa_irradiance__5061",
    "ambient_temp__5062",
    "module_temp__5063",
    "dc_voltage__5070",
    "pr__5067",
    "kwh_gross__5065",
    "reading_count",
    "data_quality",
    "T2M",
    "ALLSKY_SFC_SW_DWN",
    "CLOUD_AMT",
    "wind_gust_10m",
    "snowfall"
]

print(
    high_irradiance_negative_power[
        columns_to_check
    ]
    .sort_values("timestamp")
    .to_string(index=False)
)


# ============================================================
# 8. CLASSIFY NEGATIVE POWER RECORDS
# ============================================================

print("\n" + "=" * 70)
print("NEGATIVE POWER CLASSIFICATION")
print("=" * 70)

negative_power_classified = negative_power.copy()

negative_power_classified["irradiance_condition"] = pd.cut(
    negative_power_classified["poa_irradiance__5061"],
    bins=[-float("inf"), 1, 100, float("inf")],
    labels=[
        "Near-zero irradiance",
        "Low irradiance",
        "High irradiance"
    ]
)

print("\nNegative power records by irradiance condition:")

print(
    negative_power_classified["irradiance_condition"]
    .value_counts()
)