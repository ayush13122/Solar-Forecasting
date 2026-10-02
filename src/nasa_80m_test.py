import requests

latitude = 39.7404
longitude = -105.1719

url = "https://power.larc.nasa.gov/api/temporal/hourly/point"

params = {
    "parameters": "WS10M,WD10M",
    "community": "RE",
    "longitude": longitude,
    "latitude": latitude,
    "start": "20101231",
    "end": "20110101",
    "format": "JSON",
    "time-standard": "UTC",
    "wind-elevation": 80
}

response = requests.get(url, params=params)

print("Status Code:", response.status_code)

if response.status_code == 200:
    data = response.json()

    print("\nParameters received:")
    print(data["properties"]["parameter"].keys())

    print("\nSample values:")

    for parameter, values in data["properties"]["parameter"].items():
        print(parameter, list(values.items())[:5])

else:
    print(response.text)