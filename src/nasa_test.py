# import requests
#
# latitude = 39.7404
# longitude = -105.1719
#
# parameters = (
#     "T2M,"
#     "RH2M,"
#     "WS10M,"
#     "WD10M,"
#     "PS,"
#     "PRECTOTCORR,"
#     "ALLSKY_SFC_SW_DWN,"
#     "CLOUD_AMT"
# )
#
# url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
#
# params = {
#     "parameters": parameters,
#     "community": "RE",
#     "longitude": longitude,
#     "latitude": latitude,
#     "start": "20101231",
#     "end": "20110101",
#     "format": "JSON",
#     "time-standard": "UTC"
# }
#
# response = requests.get(url, params=params)
#
# print("Status Code:", response.status_code)
#
# if response.status_code == 200:
#     data = response.json()
#
#     print("\nNASA parameters received:")
#     print(data["properties"]["parameter"].keys())
#
# else:
#     print("\nRequest failed:")
#     print(response.text)



# import requests
# import pandas as pd
#
# latitude = 39.7404
# longitude = -105.1719
#
# parameters = "ALLSKY_SFC_SW_DWN,CLOUD_AMT"
#
# url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
#
# chunks = [
#     ("20101231", "20121231"),
#     ("20130101", "20141231"),
#     ("20150101", "20161231"),
#     ("20170101", "20180114")
# ]
#
# all_data = []
#
# for start_date, end_date in chunks:
#
#     print(f"Downloading: {start_date} → {end_date}")
#
#     params = {
#         "parameters": parameters,
#         "community": "RE",
#         "longitude": longitude,
#         "latitude": latitude,
#         "start": start_date,
#         "end": end_date,
#         "format": "JSON",
#         "time-standard": "UTC"
#     }
#
#     response = requests.get(url, params=params)
#
#     print("Status:", response.status_code)
#
#     if response.status_code == 200:
#
#         data = response.json()
#
#         parameter_data = data["properties"]["parameter"]
#
#         df = pd.DataFrame(parameter_data)
#
#         # NASA keys are YYYYMMDDHH
#         df.index = pd.to_datetime(df.index, format="%Y%m%d%H")
#
#         df.index.name = "timestamp"
#
#         all_data.append(df)
#
#     else:
#         print("ERROR:")
#         print(response.text)
#
# # Combine all chunks
# nasa_additional = pd.concat(all_data)
#
# # Remove any duplicate timestamps
# nasa_additional = nasa_additional[~nasa_additional.index.duplicated(keep="first")]
#
# print("\n============================================")
# print("NASA ADDITIONAL DATA CREATED")
# print("============================================")
#
# print("Shape:", nasa_additional.shape)
# print("First timestamp:", nasa_additional.index.min())
# print("Last timestamp:", nasa_additional.index.max())
#
# print("\nColumns:")
# print(nasa_additional.columns.tolist())
#
# print("\nFirst 5 rows:")
# print(nasa_additional.head())
#
# # Save file
# nasa_additional.to_csv("NASA_additional_hourly.csv")
#
# print("\nFile saved as: NASA_additional_hourly.csv")


import pandas as pd
import os

print("Loading main combined dataset...")

# Main PVDAQ + NASA dataset
main_file = "data/processed/PVDAQ_1433_NASA_Combined_Hourly.csv"

df = pd.read_csv(main_file)

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Set timestamp as index
df = df.set_index("timestamp")

print("Main dataset shape:", df.shape)


print("\nLoading additional NASA data...")

additional_file = "NASA_additional_hourly.csv"

nasa_additional = pd.read_csv(additional_file)

# Convert timestamp
nasa_additional["timestamp"] = pd.to_datetime(
    nasa_additional["timestamp"]
)

# Set timestamp as index
nasa_additional = nasa_additional.set_index("timestamp")

print("Additional NASA shape:", nasa_additional.shape)


print("\nAdding additional NASA features...")

# Merge using timestamp
df = df.join(
    nasa_additional,
    how="left"
)

print("\n============================================")
print("ADDITIONAL FEATURES ADDED")
print("============================================")

print("Final shape:", df.shape)

print("\nNew columns:")
print(nasa_additional.columns.tolist())

print("\nFinal columns:")
print(df.columns.tolist())


# Save updated dataset
output_file = "data/processed/PVDAQ_1433_NASA_Final_Hourly.csv"

df.to_csv(output_file)

print("\nFinal dataset saved:")
print(output_file)
