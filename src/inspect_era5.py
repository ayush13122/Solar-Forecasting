from pathlib import Path
import zipfile
import tempfile
import xarray as xr

# Plant coordinates
PLANT_LAT = 39.7404
PLANT_LON = -105.1719

zip_file = Path(
    "data/raw/ERA5/ERA5_2010_12_cloud_wind_snow.nc"
)

print("=" * 70)
print("ERA5 GRID POINT VERIFICATION")
print("=" * 70)

with tempfile.TemporaryDirectory() as temp_dir:

    temp_dir = Path(temp_dir)

    # Extract ZIP
    with zipfile.ZipFile(zip_file, "r") as z:
        z.extractall(temp_dir)

    # Open instant file
    instant_file = temp_dir / "data_stream-oper_stepType-instant.nc"

    ds = xr.open_dataset(
        instant_file,
        engine="netcdf4"
    )

    # Find nearest grid point
    nearest = ds.sel(
        latitude=PLANT_LAT,
        longitude=PLANT_LON,
        method="nearest"
    )

    print("\nPlant coordinates:")
    print(f"Latitude : {PLANT_LAT}")
    print(f"Longitude: {PLANT_LON}")

    print("\nNearest ERA5 grid point:")
    print(f"Latitude : {float(nearest.latitude.values)}")
    print(f"Longitude: {float(nearest.longitude.values)}")

    print("\nAvailable ERA5 latitudes:")
    print(ds.latitude.values)

    print("\nAvailable ERA5 longitudes:")
    print(ds.longitude.values)

    ds.close()

print("\n")
print("=" * 70)
print("VERIFICATION COMPLETED")
print("=" * 70)