import cdsapi
import os

# ============================================================
# ERA5 900 hPa WIND DATA DOWNLOADER
# System 1433 - Golden, Colorado
# Period: 2010-12 to 2018-01
# Resolution: Hourly
# ============================================================

client = cdsapi.Client()

output_dir = "data/raw/ERA5_900hPa"
os.makedirs(output_dir, exist_ok=True)

# Approximate area around System 1433
area = [40.0, -106.0, 39.0, -104.0]

# Years and months
years = list(range(2010, 2019))

for year in years:

    # December 2010 onwards
    start_month = 12 if year == 2010 else 1

    # January 2018 only
    end_month = 1 if year == 2018 else 12

    for month in range(start_month, end_month + 1):

        output_file = os.path.join(
            output_dir,
            f"ERA5_900hPa_{year}_{month:02d}.nc"
        )

        # Don't download again if already present
        if os.path.exists(output_file):
            print(f"SKIPPING {year}-{month:02d} - already downloaded")
            continue

        print("=" * 60)
        print(f"Downloading ERA5 900 hPa: {year}-{month:02d}")
        print("=" * 60)

        # Number of days in each month
        if month == 2:
            if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
                days = 29
            else:
                days = 28
        elif month in [4, 6, 9, 11]:
            days = 30
        else:
            days = 31

        request = {
            "product_type": ["reanalysis"],

            "variable": [
                "u_component_of_wind",
                "v_component_of_wind"
            ],

            "pressure_level": ["900"],

            "year": [str(year)],
            "month": [f"{month:02d}"],

            "day": [
                f"{day:02d}"
                for day in range(1, days + 1)
            ],

            "time": [
                f"{hour:02d}:00"
                for hour in range(24)
            ],

            "area": area,

            "data_format": "netcdf",
            "download_format": "unarchived"
        }

        try:

            client.retrieve(
                "reanalysis-era5-pressure-levels",
                request,
                output_file
            )

            print(f"SUCCESS: {year}-{month:02d}")

        except Exception as e:

            print(f"FAILED: {year}-{month:02d}")
            print(e)

print("=" * 60)
print("ERA5 900 hPa DOWNLOAD PROCESS COMPLETED")
print("=" * 60)