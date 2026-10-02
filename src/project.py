import pandas as pd
import os

# ============================================
# 1. File paths
# ============================================

pvdaq_path = "data/raw/PVDAQ_System_1433.csv"
nasa_path = "data/raw/NASA_weather_hourly.csv"
#
# output_path = "data/processed/PVDAQ_1433_NASA_Combined_Hourly.csv"
#
#
# # ============================================
# # 2. Load PVDAQ data
# # ============================================
#
# print("Loading PVDAQ data...")
#
df = pd.read_csv(pvdaq_path)

df["measured_on"] = pd.to_datetime(df["measured_on"])

df = df.set_index("measured_on")
#
# print("PVDAQ loaded successfully!")
# print("Original PVDAQ shape:", df.shape)
#
#
# # ============================================
# # 3. Convert PVDAQ from 15-minute to hourly
# # ============================================
#
# print("\nConverting PVDAQ data to hourly...")
#
hourly_df = df.resample("1h").agg({
    "ac_power__5069": "mean",
    "ambient_temp__5062": "mean",
    "module_temp__5063": "mean",
    "poa_irradiance__5061": "mean",
    "dc_voltage__5070": "mean",
    "pr__5067": "mean",
    "kwh_gross__5065": "sum"
})
#
# # Number of actual 15-minute AC power readings in each hour
hourly_df["reading_count"] = (
    df["ac_power__5069"]
    .resample("1h")
    .count()
)
#
# print("Hourly PVDAQ shape:", hourly_df.shape)
#
#
# # ============================================
# # 4. Load NASA weather data
# # ============================================
#
# print("\nLoading NASA weather data...")
#
nasa_df = pd.read_csv(nasa_path)
#
# nasa_df["timestamp"] = pd.to_datetime(nasa_df["timestamp"])
#
nasa_df = nasa_df.set_index("timestamp")
#
# print("NASA data loaded successfully!")
# print("NASA shape:", nasa_df.shape)
#
#
# # ============================================
# # 5. Merge PVDAQ + NASA
# # ============================================
#
# print("\nMerging PVDAQ and NASA data...")
#
# hourly_df.index.name = "timestamp"
#
# combined_df = hourly_df.join(
#     nasa_df,
#     how="inner"
# )
#
# print("Combined dataset created!")
# print("Combined shape:", combined_df.shape)
#
#
# # ============================================
# # 6. Data quality classification
# # ============================================
#
# combined_df["data_quality"] = combined_df["reading_count"].apply(
#     lambda x: "Complete"
#     if x == 4
#     else "Missing"
#     if x == 0
#     else "Incomplete"
# )
#
#
# # ============================================
# # 7. Create processed folder if needed
# # ============================================
#
# os.makedirs(
#     "data/processed",
#     exist_ok=True
# )
#
#
# # ============================================
# # 8. Save final combined CSV
# # ============================================
#
# combined_df.to_csv(output_path)
#
# print("\n============================================")
# print("FINAL DATASET CREATED SUCCESSFULLY!")
# print("============================================")
#
# print("File:", output_path)
# print("Shape:", combined_df.shape)
#
# print("\nData Quality:")
# print(combined_df["data_quality"].value_counts())
#
# print("\nFirst 5 rows:")
# print(combined_df.head())



extra_nasa = nasa_df.index.difference(hourly_df.index)

print("NASA-only timestamps:")
print(extra_nasa)
