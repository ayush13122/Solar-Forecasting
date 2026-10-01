# Final Dataset Quality Audit

**Status:** Ready for EDA  
**Source:** `PVDAQ_1433_NASA_ERA5_900hPa_Final_Hourly.csv`  
**Audit generated (UTC):** 2026-10-01T10:43:24.194319+00:00

## Dataset identity

| Check | Result |
| --- | --- |
| Rows | 61,726 (expected 61,726) |
| Columns | 25 (expected 25) |
| Time range | 2010-12-31T00:00:00 to 2018-01-14T21:00:00 |
| Duplicate timestamps | 0 |
| Missing hourly timestamps | 0 |
| Schema differences | 0 missing, 0 unexpected |

## PVDAQ target: `ac_power__5069`

| Check | Result |
| --- | --- |
| Available observations | 56,985 |
| Missing observations | 4,741 |
| Minimum / maximum kW | -0.75 / 383.616 |
| Mean / median kW | 64.4399 / 0.0 |
| Zero or negative observations | 29,306 |
| Negative observations | 1,267 |

Zero and slightly negative generation values are retained for now; they commonly
occur at night or can reflect sensor noise. We will make any modelling filter
explicit in the next step instead of silently changing the source data.

## `data_quality` labels

| Check | Result |
| --- | --- |
| Complete | 54,707 |
| Missing | 4,741 |
| Incomplete | 2,278 |

## Missing values by column

| Check | Result |
| --- | --- |
| ac_power__5069 | 4,741 (7.681%) |
| ambient_temp__5062 | 1,659 (2.688%) |
| module_temp__5063 | 1,659 (2.688%) |
| poa_irradiance__5061 | 1,659 (2.688%) |
| dc_voltage__5070 | 46,490 (75.317%) |
| pr__5067 | 33,141 (53.691%) |

## Next approved project step

Perform exploratory data analysis from this verified master CSV: inspect power
generation over time, sunlight/irradiance relationships, weather correlations,
and missing-data patterns. The raw PVDAQ and final master CSVs remain unchanged.
