import cdsapi

client = cdsapi.Client()

dataset = "reanalysis-era5-pressure-levels"

request = {
    "product_type": ["reanalysis"],
    "variable": [
        "u_component_of_wind",
        "v_component_of_wind"
    ],
    "pressure_level": ["900"],
    "year": ["2017"],
    "month": ["08"],
    "day": ["01"],
    "time": ["00:00"],
    "data_format": "netcdf",
    "download_format": "unarchived"
}

print("Checking ERA5 900 hPa availability...")

try:
    client.retrieve(
        dataset,
        request,
        "data/raw/ERA5/test_900hpa.nc"
    )

    print("SUCCESS: 900 hPa U/V wind data is available.")

except Exception as e:
    print("ERROR: 900 hPa data is not available.")
    print(e)