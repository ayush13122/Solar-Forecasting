import pandas as pd

df = pd.read_csv(
    "data/processed/PVDAQ_1433_NASA_ERA5_900hPa_Final_Hourly.csv"
)

df["timestamp"] = pd.to_datetime(df["timestamp"])

print("=" * 70)
print("DUPLICATE ANALYSIS")
print("=" * 70)

print(f"\nTotal rows: {len(df):,}")

print(f"\nDuplicate complete rows: {df.duplicated().sum():,}")

print(f"\nDuplicate timestamps: {df['timestamp'].duplicated().sum():,}")