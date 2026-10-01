# Exploratory Data Analysis — PVDAQ System 1433

**Source:** `PVDAQ_1433_NASA_ERA5_900hPa_Final_Hourly.csv`  
**Analysis rule:** `data_quality == Complete` and non-missing `ac_power__5069`

## Data used for analysis

| Metric | Value |
| --- | --- |
| Hourly rows in master data | 61,726 |
| Complete target observations | 54,707 (88.629%) |
| Daylight rows for irradiance analysis | 25,840 |
| Mean AC power across complete rows | 65.1246 kW |
| Maximum AC power | 376.0 kW |

## What the figures show

1. [Monthly generation and availability](figures/01_monthly_generation_and_availability.png) separates changes in observed generation from gaps in the PVDAQ readings.
2. [Typical daily generation profile](figures/02_typical_daily_generation_profile.png) checks whether the plant follows the expected solar-shaped day cycle.
3. [Irradiance versus AC power](figures/03_irradiance_vs_ac_power.png) is a diagnostic relationship using an on-site plant sensor.
4. [Weather-feature correlations](figures/04_weather_feature_correlations.png) ranks the external-weather relationships independently of that sensor.

## Initial evidence

- The plane-of-array irradiance / AC-power Pearson correlation is **0.9533** across complete hours. This is expected physically, but it is a same-time measurement—not a future-known feature.
- The strongest external-weather correlations are shown below. Correlation identifies association only; it does not prove causality or choose the final model.

| Metric | Value |
| --- | --- |
| RH2M | -0.6873 |
| T2M | 0.5709 |
| CLOUD_AMT | -0.2581 |
| wind_direction_900_mb | 0.251 |
| medium_cloud_cover | -0.2372 |
| WD10M | -0.1913 |
| wind_speed_900_mb | 0.1787 |
| high_cloud_cover | -0.1419 |

## Modelling guardrails

- Missing target rows are excluded from training and preserved in the source data.
- `dc_voltage__5070` and `pr__5067` are not baseline model features because their missingness is substantial and they are operational measurements, not weather forecasts.
- We will use chronological, not random, train/test splits. That better reflects a real prediction scenario.
- A same-timestamp model will be called **regression/nowcasting**. A future forecast will need lag features and weather known before the prediction time.
