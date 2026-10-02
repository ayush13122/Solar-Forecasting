# import cdsapi
# import os
#
# client = cdsapi.Client()
#
# output_dir = "data/raw/ERA5"
# os.makedirs(output_dir, exist_ok=True)
#
# client.retrieve(
#     "reanalysis-era5-single-levels",
#     {
#         "product_type": "reanalysis",
#
#         "variable": [
#             "high_cloud_cover",
#             "medium_cloud_cover",
#             "low_cloud_cover",
#             "10m_wind_gust_since_previous_post_processing",
#             "snowfall"
#         ],
#
#         "year": "2018",
#         "month": "01",
#
#         "day": [
#             "01", "02", "03", "04", "05",
#             "06", "07", "08", "09", "10",
#             "11", "12", "13", "14", "15",
#             "16", "17", "18", "19", "20",
#             "21", "22", "23", "24", "25",
#             "26", "27", "28", "29", "30", "31"
#         ],
#
#         "time": [
#             "00:00", "01:00", "02:00", "03:00",
#             "04:00", "05:00", "06:00", "07:00",
#             "08:00", "09:00", "10:00", "11:00",
#             "12:00", "13:00", "14:00", "15:00",
#             "16:00", "17:00", "18:00", "19:00",
#             "20:00", "21:00", "22:00", "23:00"
#         ],
#
#         "area": [
#             40.0,
#             -106.0,
#             39.0,
#             -104.0
#         ],
#
#         "data_format": "netcdf",
#         "download_format": "unarchived"
#     },
#     os.path.join(output_dir, "ERA5_2018_01_cloud_wind_snow.nc")
# )
#
# print("January 2018 ERA5 download successful!")

import cdsapi
import os
import calendar

# CDS client
client = cdsapi.Client()

# Output folder
output_dir = "data/raw/ERA5"
os.makedirs(output_dir, exist_ok=True)

# Years and months
start_year = 2010
start_month = 12

end_year = 2018
end_month = 1

# All required variables
variables = [
    "high_cloud_cover",
    "medium_cloud_cover",
    "low_cloud_cover",
    "10m_wind_gust_since_previous_post_processing",
    "snowfall"
]

# All hours
times = [
    "00:00", "01:00", "02:00", "03:00",
    "04:00", "05:00", "06:00", "07:00",
    "08:00", "09:00", "10:00", "11:00",
    "12:00", "13:00", "14:00", "15:00",
    "16:00", "17:00", "18:00", "19:00",
    "20:00", "21:00", "22:00", "23:00"
]

# Small area around System 1433
area = [
    40.0,      # North
    -106.0,    # West
    39.0,      # South
    -104.0     # East
]

for year in range(start_year, end_year + 1):

    first_month = start_month if year == start_year else 1
    last_month = end_month if year == end_year else 12

    for month in range(first_month, last_month + 1):

        month_str = f"{month:02d}"

        output_file = os.path.join(
            output_dir,
            f"ERA5_{year}_{month_str}_cloud_wind_snow.nc"
        )

        # Skip if already downloaded
        if os.path.exists(output_file):
            print(f"SKIPPING {year}-{month_str} - already downloaded")
            continue

        # Number of days in month
        days_in_month = calendar.monthrange(year, month)[1]

        days = [f"{day:02d}" for day in range(1, days_in_month + 1)]

        print("=" * 60)
        print(f"Downloading ERA5: {year}-{month_str}")
        print("=" * 60)

        try:

            client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",

                    "variable": variables,

                    "year": str(year),
                    "month": month_str,

                    "day": days,

                    "time": times,

                    "area": area,

                    "data_format": "netcdf",
                    "download_format": "unarchived"
                },

                output_file
            )

            print(f"SUCCESS: {year}-{month_str}")

        except Exception as e:

            print(f"FAILED: {year}-{month_str}")
            print(f"Error: {e}")

            # Continue with next month
            continue


print("=" * 60)
print("ERA5 DOWNLOAD PROCESS COMPLETED")
print("=" * 60)