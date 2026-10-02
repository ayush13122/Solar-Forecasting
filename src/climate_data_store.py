# import cdsapi
#
# client = cdsapi.Client()
#
# print("CDS API connection successful!")

# import cdsapi
#
# client = cdsapi.Client()
#
# client.retrieve(
#     "reanalysis-era5-single-levels",
#     {
#         "product_type": "reanalysis",
#         "variable": [
#             "2m_temperature"
#         ],
#         "year": "2018",
#         "month": "01",
#         "day": [
#             "14"
#         ],
#         "time": [
#             "12:00"
#         ],
#         "data_format": "netcdf",
#         "download_format": "unarchived"
#     },
#     "era5_test.nc"
# )
#
# print("ERA5 test download successful!")



import cdsapi

client = cdsapi.Client()

client.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",

        "variable": [
            "high_cloud_cover",
            "medium_cloud_cover",
            "low_cloud_cover",
            "10m_wind_gust_since_previous_post_processing",
            "snowfall"
        ],

        "year": "2018",
        "month": "01",
        "day": ["14"],

        "time": [
            "12:00"
        ],

        "area": [
            40.0,       # North
            -106.0,     # West
            39.0,       # South
            -104.0      # East
        ],

        "data_format": "netcdf",
        "download_format": "unarchived"
    },
    "era5_variables_test.nc"
)

print("ERA5 required-variables test successful!")
