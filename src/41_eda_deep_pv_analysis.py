# ================================================================
# STEP 77 - DEEP PV EDA
# ================================================================
# Deep exploratory analysis for:
# Solar Power Generation Forecasting & Plant Performance Analytics
#
# This script:
#   - does NOT modify source datasets
#   - does NOT train ML models
#   - does NOT evaluate the 2017-2018 test model
#
# Outputs:
#   reports/eda/deep_analysis/
# ================================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


print("=" * 70)
print("STEP 77 - DEEP PV EDA")
print("=" * 70)


# ----------------------------------------------------------------
# 1. FIND ACTUAL PROJECT ROOT
# ----------------------------------------------------------------

current_path = Path(__file__).resolve()

PROJECT_ROOT = None

KNOWN_FILES = [
    "PVDAQ_1433_ML_Ready_Preprocessed_Hourly.csv",
    "PVDAQ_1433_NASA_ERA5_900hPa_Final_Hourly.csv",
]

for parent in current_path.parents:

    processed_dir = (
            parent
            / "data"
            / "processed"
    )

    if not processed_dir.exists():
        continue

    ml_ready_exists = (
            processed_dir
            / KNOWN_FILES[0]
    ).exists()

    master_exists = (
            processed_dir
            / KNOWN_FILES[1]
    ).exists()

    has_venv = (
            (parent / ".venv").exists()
            or
            (parent / ".venv_tf").exists()
    )

    if has_venv and (
            ml_ready_exists
            or master_exists
    ):
        PROJECT_ROOT = parent
        break


if PROJECT_ROOT is None:

    raise FileNotFoundError(
        "Actual project root not found."
    )


print(
    f"\nActual project root:\n{PROJECT_ROOT}"
)


# ----------------------------------------------------------------
# 2. PATHS
# ----------------------------------------------------------------

PROCESSED_DIR = (
        PROJECT_ROOT
        / "data"
        / "processed"
)

EDA_DIR = (
        PROJECT_ROOT
        / "reports"
        / "eda"
        / "deep_analysis"
)

EDA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


ML_READY_FILE = (
        PROCESSED_DIR
        / "PVDAQ_1433_ML_Ready_Preprocessed_Hourly.csv"
)

MASTER_FILE = (
        PROCESSED_DIR
        / "PVDAQ_1433_NASA_ERA5_900hPa_Final_Hourly.csv"
)


# ----------------------------------------------------------------
# 3. LOAD DATA
# ----------------------------------------------------------------

if not ML_READY_FILE.exists():

    raise FileNotFoundError(
        f"ML-ready file not found:\n{ML_READY_FILE}"
    )


if not MASTER_FILE.exists():

    raise FileNotFoundError(
        f"Master file not found:\n{MASTER_FILE}"
    )


ml_df = pd.read_csv(
    ML_READY_FILE,
    parse_dates=["timestamp"]
)

master_df = pd.read_csv(
    MASTER_FILE,
    parse_dates=["timestamp"]
)


ml_df = (
    ml_df
    .sort_values("timestamp")
    .reset_index(drop=True)
)

master_df = (
    master_df
    .sort_values("timestamp")
    .reset_index(drop=True)
)


print(
    f"\nML-ready dataset : {ml_df.shape}"
)

print(
    f"Master dataset   : {master_df.shape}"
)

print(
    f"\nML date range:\n"
    f"{ml_df['timestamp'].min()} → "
    f"{ml_df['timestamp'].max()}"
)

print(
    f"\nMaster date range:\n"
    f"{master_df['timestamp'].min()} → "
    f"{master_df['timestamp'].max()}"
)


# ----------------------------------------------------------------
# 4. COMMON COLUMN CHECK
# ----------------------------------------------------------------

TARGET = "ac_power__5069"

POA = "poa_irradiance__5061"

AMBIENT = "ambient_temp__5062"

MODULE = "module_temp__5063"

GROSS_ENERGY = "kwh_gross__5065"


required_ml = [
    TARGET,
    POA,
    AMBIENT,
    MODULE,
]

required_master = [
    TARGET,
    POA,
    AMBIENT,
    MODULE,
    GROSS_ENERGY,
]


for col in required_ml:

    if col not in ml_df.columns:

        raise ValueError(
            f"Required ML-ready column missing: {col}"
        )


for col in required_master:

    if col not in master_df.columns:

        raise ValueError(
            f"Required master column missing: {col}"
        )


# ----------------------------------------------------------------
# 5. CREATE TIME FEATURES
# ----------------------------------------------------------------

for data in [ml_df, master_df]:

    data["year"] = (
        data["timestamp"].dt.year
    )

    data["month"] = (
        data["timestamp"].dt.month
    )

    data["hour"] = (
        data["timestamp"].dt.hour
    )

    data["day_of_year"] = (
        data["timestamp"].dt.dayofyear
    )


# ----------------------------------------------------------------
# 6. BASIC DEEP EDA SUMMARY
# ----------------------------------------------------------------

basic_summary = pd.DataFrame({

    "dataset": [
        "ML-ready",
        "Master",
    ],

    "rows": [
        len(ml_df),
        len(master_df),
    ],

    "columns": [
        len(ml_df.columns),
        len(master_df.columns),
    ],

    "start": [
        ml_df["timestamp"].min(),
        master_df["timestamp"].min(),
    ],

    "end": [
        ml_df["timestamp"].max(),
        master_df["timestamp"].max(),
    ],

    "duplicate_timestamps": [
        ml_df["timestamp"].duplicated().sum(),
        master_df["timestamp"].duplicated().sum(),
    ],

})


basic_summary.to_csv(
    EDA_DIR
    / "01_deep_eda_basic_summary.csv",
    index=False
)


# ----------------------------------------------------------------
# 7. HOURLY × MONTHLY GENERATION HEATMAP
# ----------------------------------------------------------------

print(
    "\nCreating Hour × Month generation heatmap..."
)


hour_month = (
    ml_df
    .pivot_table(
        index="hour",
        columns="month",
        values=TARGET,
        aggfunc="mean"
    )
)


hour_month.to_csv(
    EDA_DIR
    / "02_hour_month_ac_power_profile.csv"
)


plt.figure(
    figsize=(11, 7)
)

plt.imshow(
    hour_month.values,
    aspect="auto",
    origin="lower"
)

plt.colorbar(
    label="Mean AC Power (kW)"
)

plt.xlabel(
    "Month"
)

plt.ylabel(
    "Hour of Day"
)

plt.title(
    "Mean AC Power by Hour and Month"
)

plt.xticks(
    range(12),
    range(1, 13)
)

plt.yticks(
    range(24),
    range(24)
)

plt.tight_layout()

plt.savefig(
    EDA_DIR
    / "02_hour_month_ac_power_heatmap.png",
    dpi=200
)

plt.close()


# ----------------------------------------------------------------
# 8. AC AUTOCORRELATION
# ----------------------------------------------------------------

print(
    "\nCalculating AC power autocorrelation..."
)


# Use full hourly master so timestamp spacing is preserved.
ac_series = (
    master_df
    .set_index("timestamp")[TARGET]
    .sort_index()
)


autocorrelation_rows = []


for lag in [1, 2, 3, 6, 12, 24, 48, 72]:

    lagged = (
        pd.concat(
            [
                ac_series.rename("current"),
                ac_series.shift(lag).rename("lagged"),
            ],
            axis=1,
        )
        .dropna()
    )


    correlation = (
        lagged["current"]
        .corr(
            lagged["lagged"]
        )
    )


    autocorrelation_rows.append({

        "lag_hours": lag,

        "correlation": correlation,

        "paired_observations":
            len(lagged),
    })


autocorrelation_df = pd.DataFrame(
    autocorrelation_rows
)


autocorrelation_df.to_csv(
    EDA_DIR
    / "03_ac_autocorrelation.csv",
    index=False
)


print(
    autocorrelation_df.to_string(
        index=False
    )
)


plt.figure(
    figsize=(10, 6)
)

plt.plot(
    autocorrelation_df["lag_hours"],
    autocorrelation_df["correlation"],
    marker="o"
)

plt.xlabel(
    "Lag (hours)"
)

plt.ylabel(
    "Correlation"
)

plt.title(
    "AC Power Autocorrelation"
)

plt.xticks(
    autocorrelation_df["lag_hours"]
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    EDA_DIR
    / "03_ac_autocorrelation.png",
    dpi=200
)

plt.close()


# ----------------------------------------------------------------
# 9. AC RAMP-RATE ANALYSIS
# ----------------------------------------------------------------

print(
    "\nAnalyzing hourly AC power ramps..."
)


ramp_df = master_df[
    [
        "timestamp",
        TARGET,
    ]
].copy()


ramp_df[
    "ac_change_1h"
] = ramp_df[
    TARGET
].diff()


ramp_df[
    "absolute_ac_change_1h"
] = (
    ramp_df[
        "ac_change_1h"
    ].abs()
)


valid_ramps = ramp_df[
    "ac_change_1h"
].dropna()


ramp_summary = pd.DataFrame({

    "metric": [

        "mean_change",

        "median_change",

        "std_change",

        "min_change",

        "max_change",

        "mean_absolute_change",

        "95th_percentile_absolute_change",

        "99th_percentile_absolute_change",
    ],

    "value": [

        valid_ramps.mean(),

        valid_ramps.median(),

        valid_ramps.std(),

        valid_ramps.min(),

        valid_ramps.max(),

        valid_ramps.abs().mean(),

        valid_ramps.abs().quantile(0.95),

        valid_ramps.abs().quantile(0.99),
    ],
})


ramp_summary.to_csv(
    EDA_DIR
    / "04_ramp_summary.csv",
    index=False
)


print(
    ramp_summary.to_string(
        index=False
    )
)


# Top absolute ramp events
top_ramps = (
    ramp_df
    .dropna(
        subset=["absolute_ac_change_1h"]
    )
    .sort_values(
        "absolute_ac_change_1h",
        ascending=False
    )
    .head(50)
)


top_ramps.to_csv(
    EDA_DIR
    / "04_top_50_ramp_events.csv",
    index=False
)


plt.figure(
    figsize=(10, 6)
)

plt.hist(
    valid_ramps,
    bins=80
)

plt.xlabel(
    "1-hour AC Power Change (kW)"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "Distribution of Hourly AC Power Ramps"
)

plt.tight_layout()

plt.savefig(
    EDA_DIR
    / "04_ac_ramp_distribution.png",
    dpi=200
)

plt.close()


# ----------------------------------------------------------------
# 10. IRRADIANCE BIN ANALYSIS
# ----------------------------------------------------------------

print(
    "\nAnalyzing AC response across POA irradiance bins..."
)


irradiance_df = ml_df[
    [
        POA,
        TARGET,
    ]
].dropna().copy()


irradiance_bins = [

    -np.inf,
    0,
    100,
    200,
    400,
    600,
    800,
    1000,
    np.inf,
]


irradiance_labels = [

    "<=0",

    "0-100",

    "100-200",

    "200-400",

    "400-600",

    "600-800",

    "800-1000",

    ">1000",
]


irradiance_df[
    "poa_bin"
] = pd.cut(
    irradiance_df[POA],
    bins=irradiance_bins,
    labels=irradiance_labels,
    include_lowest=True
)


irradiance_profile = (
    irradiance_df
    .groupby(
        "poa_bin",
        observed=False
    )[TARGET]
    .agg(
        [
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max",
        ]
    )
    .reset_index()
)


irradiance_profile.to_csv(
    EDA_DIR
    / "05_irradiance_bin_analysis.csv",
    index=False
)


print(
    irradiance_profile.to_string(
        index=False
    )
)


plt.figure(
    figsize=(11, 6)
)

plt.bar(
    irradiance_profile["poa_bin"].astype(str),
    irradiance_profile["mean"]
)

plt.xlabel(
    "POA Irradiance Bin"
)

plt.ylabel(
    "Mean AC Power (kW)"
)

plt.title(
    "AC Power Response Across POA Irradiance Bins"
)

plt.xticks(
    rotation=30
)

plt.tight_layout()

plt.savefig(
    EDA_DIR
    / "05_irradiance_bin_response.png",
    dpi=200
)

plt.close()


# ----------------------------------------------------------------
# 11. CLOUD COVER BIN ANALYSIS
# ----------------------------------------------------------------

if "CLOUD_AMT" in ml_df.columns:

    print(
        "\nAnalyzing cloud-cover regimes..."
    )


    cloud_df = ml_df[
        [
            "CLOUD_AMT",
            TARGET,
            POA,
        ]
    ].dropna().copy()


    cloud_bins = [
        -np.inf,
        20,
        40,
        60,
        80,
        100,
        np.inf,
    ]


    cloud_labels = [

        "0-20",

        "20-40",

        "40-60",

        "60-80",

        "80-100",

        ">100",
    ]


    cloud_df[
        "cloud_bin"
    ] = pd.cut(
        cloud_df[
            "CLOUD_AMT"
        ],
        bins=cloud_bins,
        labels=cloud_labels,
        include_lowest=True
    )


    cloud_profile = (
        cloud_df
        .groupby(
            "cloud_bin",
            observed=False
        )
        .agg(
            mean_ac_power=(
                TARGET,
                "mean"
            ),
            median_ac_power=(
                TARGET,
                "median"
            ),
            mean_poa=(
                POA,
                "mean"
            ),
            count=(
                TARGET,
                "count"
            ),
        )
        .reset_index()
    )


    cloud_profile.to_csv(
        EDA_DIR
        / "06_cloud_bin_analysis.csv",
        index=False
    )


    print(
        cloud_profile.to_string(
            index=False
        )
    )


# ----------------------------------------------------------------
# 12. TEMPERATURE RELATIONSHIP
# ----------------------------------------------------------------

print(
    "\nAnalyzing temperature relationships..."
)


temperature_df = ml_df[
    [
        AMBIENT,
        MODULE,
        POA,
        TARGET,
    ]
].dropna().copy()


temperature_df[
    "module_ambient_delta"
] = (
        temperature_df[
            MODULE
        ]
        - temperature_df[
            AMBIENT
        ]
)


temperature_summary = pd.DataFrame({

    "metric": [

        "ambient_mean",

        "ambient_median",

        "ambient_std",

        "module_mean",

        "module_median",

        "module_std",

        "module_minus_ambient_mean",

        "module_minus_ambient_median",

        "module_minus_ambient_std",
    ],

    "value": [

        temperature_df[
            AMBIENT
        ].mean(),

        temperature_df[
            AMBIENT
        ].median(),

        temperature_df[
            AMBIENT
        ].std(),

        temperature_df[
            MODULE
        ].mean(),

        temperature_df[
            MODULE
        ].median(),

        temperature_df[
            MODULE
        ].std(),

        temperature_df[
            "module_ambient_delta"
        ].mean(),

        temperature_df[
            "module_ambient_delta"
        ].median(),

        temperature_df[
            "module_ambient_delta"
        ].std(),
    ],
})


temperature_summary.to_csv(
    EDA_DIR
    / "07_temperature_summary.csv",
    index=False
)


# Module minus ambient vs POA
plt.figure(
    figsize=(10, 6)
)


sample_temperature = (
    temperature_df
    .sample(
        min(
            10000,
            len(temperature_df)
        ),
        random_state=42
    )
)


plt.scatter(
    sample_temperature[
        POA
    ],
    sample_temperature[
        "module_ambient_delta"
    ],
    s=8,
    alpha=0.25
)


plt.xlabel(
    "POA Irradiance"
)

plt.ylabel(
    "Module - Ambient Temperature (deg C)"
)

plt.title(
    "Module-Ambient Temperature Difference vs POA"
)

plt.tight_layout()

plt.savefig(
    EDA_DIR
    / "07_module_ambient_delta_vs_poa.png",
    dpi=200
)

plt.close()


# ----------------------------------------------------------------
# 13. DAILY ENERGY ANALYSIS
# ----------------------------------------------------------------

print(
    "\nAnalyzing daily energy generation..."
)


energy_time = (
    master_df
    .set_index("timestamp")
    .sort_index()
)


daily_energy = (
    energy_time[
        GROSS_ENERGY
    ]
    .resample("D")
    .sum(
        min_count=1
    )
)


daily_energy_df = (
    daily_energy
    .reset_index()
)


daily_energy_df.columns = [
    "date",
    "daily_gross_energy"
]


daily_energy_df.to_csv(
    EDA_DIR
    / "08_daily_gross_energy.csv",
    index=False
)


annual_energy = (
    energy_time[
        GROSS_ENERGY
    ]
    .resample("YS")
    .sum(
        min_count=1
    )
)


annual_energy_df = (
    annual_energy
    .reset_index()
)


annual_energy_df.columns = [
    "year",
    "annual_gross_energy"
]


annual_energy_df[
    "year"
] = annual_energy_df[
    "year"
].dt.year


annual_energy_df.to_csv(
    EDA_DIR
    / "08_annual_gross_energy.csv",
    index=False
)


monthly_energy = (
    energy_time[
        GROSS_ENERGY
    ]
    .resample("MS")
    .sum(
        min_count=1
    )
)


monthly_energy_df = (
    monthly_energy
    .reset_index()
)


monthly_energy_df.columns = [
    "month",
    "monthly_gross_energy"
]


monthly_energy_df.to_csv(
    EDA_DIR
    / "08_monthly_gross_energy.csv",
    index=False
)


print(
    "\nAnnual gross energy:"
)

print(
    annual_energy_df.to_string(
        index=False
    )
)


plt.figure(
    figsize=(11, 6)
)

plt.plot(
    annual_energy_df[
        "year"
    ],
    annual_energy_df[
        "annual_gross_energy"
    ],
    marker="o"
)

plt.xlabel(
    "Year"
)

plt.ylabel(
    "Gross Energy"
)

plt.title(
    "Annual Gross Energy Generation"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

output_file = (EDA_DIR / "08_annual_gross_energy.png").resolve()

plt.savefig(
    str(output_file),
    dpi=200,
    bbox_inches="tight",
    format="png"
)

plt.close()


# ----------------------------------------------------------------
# 14. DAILY ENERGY DISTRIBUTION
# ----------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.hist(
    daily_energy_df[
        "daily_gross_energy"
    ].dropna(),
    bins=60
)

plt.xlabel(
    "Daily Gross Energy"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "Daily Gross Energy Distribution"
)

plt.tight_layout()

plt.savefig(
    EDA_DIR
    / "08_daily_energy_distribution.png",
    dpi=200
)

plt.close()


# ----------------------------------------------------------------
# 15. AC UTILIZATION RELATIVE TO DC CAPACITY
# ----------------------------------------------------------------
# System 1433 nominal DC capacity used for contextual analysis:
# approximately 449.28 kW.
#
# This is a utilization-style diagnostic, NOT an inverter clipping
# assumption.

DC_CAPACITY_KW = 449.28


utilization_df = ml_df[
    [
        "timestamp",
        TARGET,
    ]
].dropna().copy()


utilization_df[
    "ac_utilization"
] = (
        utilization_df[
            TARGET
        ]
        / DC_CAPACITY_KW
)


utilization_summary = pd.DataFrame({

    "metric": [

        "mean_utilization",

        "median_utilization",

        "std_utilization",

        "max_utilization",

        "95th_percentile",

        "99th_percentile",
    ],

    "value": [

        utilization_df[
            "ac_utilization"
        ].mean(),

        utilization_df[
            "ac_utilization"
        ].median(),

        utilization_df[
            "ac_utilization"
        ].std(),

        utilization_df[
            "ac_utilization"
        ].max(),

        utilization_df[
            "ac_utilization"
        ].quantile(
            0.95
        ),

        utilization_df[
            "ac_utilization"
        ].quantile(
            0.99
        ),
    ],
})


utilization_summary.to_csv(
    EDA_DIR
    / "09_ac_capacity_utilization.csv",
    index=False
)


print(
    "\nAC capacity utilization summary:"
)

print(
    utilization_summary.to_string(
        index=False
    )
)


# ----------------------------------------------------------------
# 16. kWh GROSS vs AC POWER CONSISTENCY
# ----------------------------------------------------------------
# For hourly data:
#
# approximate hourly energy expected from mean AC power:
#
#   AC power (kW) × 1 hour = kWh
#
# This is a consistency diagnostic only.
# It does NOT assume perfect sensor identity.

consistency_df = master_df[
    [
        "timestamp",
        TARGET,
        GROSS_ENERGY,
        POA,
    ]
].copy()


consistency_df[
    "energy_from_ac_estimate"
] = (
    consistency_df[
        TARGET
    ]
)


valid_consistency = (
                            consistency_df[
                                TARGET
                            ] > 5
                    ) & (
                            consistency_df[
                                GROSS_ENERGY
                            ] > 0
                    )


consistency_df[
    "energy_ratio"
] = np.nan


consistency_df.loc[
    valid_consistency,
    "energy_ratio"
] = (
        consistency_df.loc[
            valid_consistency,
            GROSS_ENERGY
        ]
        /
        consistency_df.loc[
            valid_consistency,
            "energy_from_ac_estimate"
        ]
)


ratio = (
    consistency_df[
        "energy_ratio"
    ]
    .dropna()
)


energy_consistency_summary = pd.DataFrame({

    "metric": [

        "valid_rows",

        "ratio_mean",

        "ratio_median",

        "ratio_std",

        "ratio_10th_percentile",

        "ratio_90th_percentile",

        "ratio_min",

        "ratio_max",
    ],

    "value": [

        len(ratio),

        ratio.mean(),

        ratio.median(),

        ratio.std(),

        ratio.quantile(
            0.10
        ),

        ratio.quantile(
            0.90
        ),

        ratio.min(),

        ratio.max(),
    ],
})


energy_consistency_summary.to_csv(
    EDA_DIR
    / "10_kwh_gross_ac_consistency.csv",
    index=False
)


print(
    "\nkWh gross / AC-power consistency:"
)

print(
    energy_consistency_summary.to_string(
        index=False
    )
)


# ----------------------------------------------------------------
# 17. DAYLIGHT VS NIGHT BEHAVIOR
# ----------------------------------------------------------------

if "solar_zenith_angle" in ml_df.columns:

    ml_df[
        "eda_daylight"
    ] = (
            ml_df[
                "solar_zenith_angle"
            ] < 90
    ).astype(int)

else:

    ml_df[
        "eda_daylight"
    ] = (
            ml_df[
                POA
            ] > 0
    ).astype(int)


daylight_summary = (
    ml_df
    .groupby(
        "eda_daylight"
    )[TARGET]
    .agg(
        [
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max",
        ]
    )
    .reset_index()
)


daylight_summary[
    "period"
] = daylight_summary[
    "eda_daylight"
].map(
    {
        0: "Night",
        1: "Daylight",
    }
)


daylight_summary.to_csv(
    EDA_DIR
    / "11_daylight_vs_night.csv",
    index=False
)


print(
    "\nDaylight vs night:"
)

print(
    daylight_summary.to_string(
        index=False
    )
)


# ----------------------------------------------------------------
# 18. SENSOR RELATIONSHIP CORRELATIONS
# ----------------------------------------------------------------

sensor_columns = [

    TARGET,

    POA,

    AMBIENT,

    MODULE,

]


sensor_columns = [
    col
    for col in sensor_columns
    if col in ml_df.columns
]


sensor_corr = (
    ml_df[
        sensor_columns
    ]
    .corr()
)


sensor_corr.to_csv(
    EDA_DIR
    / "12_sensor_relationship_correlation.csv"
)


print(
    "\nSensor relationship correlation:"
)

print(
    sensor_corr.to_string()
)


# ----------------------------------------------------------------
# 19. CORE SENSOR DATA-QUALITY PATTERNS
# ----------------------------------------------------------------
# Use MASTER because ML-ready data has already excluded/processed
# unavailable target/core rows.

quality_columns = [

    "ac_power__5069",

    "ambient_temp__5062",

    "module_temp__5063",

    "poa_irradiance__5061",

    "dc_voltage__5070",

    "pr__5067",

    "reading_count",

]


available_quality_columns = [
    col
    for col in quality_columns
    if col in master_df.columns
]


if len(available_quality_columns) > 0:

    print(
        "\nAnalyzing original master data-quality patterns..."
    )


    yearly_quality_rows = []


    for year, group in master_df.groupby(
            "year"
    ):

        row = {
            "year": year
        }


        for col in available_quality_columns:

            row[
                f"{col}_missing_pct"
            ] = (
                    group[
                        col
                    ].isna().mean()
                    * 100
            )


        if "reading_count" in group.columns:

            row[
                "complete_hour_pct"
            ] = (
                    (
                            group[
                                "reading_count"
                            ] == 4
                    )
                    .mean()
                    * 100
            )


            row[
                "zero_reading_hour_pct"
            ] = (
                    (
                            group[
                                "reading_count"
                            ] == 0
                    )
                    .mean()
                    * 100
            )


        yearly_quality_rows.append(
            row
        )


    yearly_quality_df = pd.DataFrame(
        yearly_quality_rows
    )


    yearly_quality_df.to_csv(
        EDA_DIR
        / "13_yearly_data_quality_patterns.csv",
        index=False
    )


    print(
        yearly_quality_df.to_string(
            index=False
        )
    )


# ----------------------------------------------------------------
# 20. MISSINGNESS BY MONTH
# ----------------------------------------------------------------

missingness_columns = [

    TARGET,

    AMBIENT,

    MODULE,

    POA,

    "dc_voltage__5070",

    "pr__5067",
]


available_missingness_columns = [
    col
    for col in missingness_columns
    if col in master_df.columns
]


if len(available_missingness_columns) > 0:

    monthly_missing_rows = []


    for month, group in master_df.groupby(
            "month"
    ):

        row = {
            "month": month
        }


        for col in available_missingness_columns:

            row[
                f"{col}_missing_pct"
            ] = (
                    group[
                        col
                    ].isna().mean()
                    * 100
            )


        monthly_missing_rows.append(
            row
        )


    monthly_missing_df = pd.DataFrame(
        monthly_missing_rows
    )


    monthly_missing_df.to_csv(
        EDA_DIR
        / "14_monthly_missingness_patterns.csv",
        index=False
    )


# ----------------------------------------------------------------
# 21. TOP CORRELATIONS WITH AC POWER
# ----------------------------------------------------------------

numeric_columns = ml_df.select_dtypes(
    include=np.number
).columns.tolist()


correlation_with_target = (
    ml_df[
        numeric_columns
    ]
    .corr()[TARGET]
    .sort_values(
        ascending=False
    )
)


correlation_with_target = (
    correlation_with_target
    .to_frame(
        name="correlation_with_ac_power"
    )
)


correlation_with_target[
    "absolute_correlation"
] = (
    correlation_with_target[
        "correlation_with_ac_power"
    ].abs()
)


correlation_with_target.to_csv(
    EDA_DIR
    / "15_all_target_correlations.csv"
)


print(
    "\nTop correlations with AC power:"
)

print(
    correlation_with_target.head(
        15
    ).to_string()
)


# ----------------------------------------------------------------
# 22. LOW / HIGH POA OPERATING REGIMES
# ----------------------------------------------------------------

regime_df = ml_df[
    [
        POA,
        TARGET,
    ]
].dropna().copy()


regime_df[
    "regime"
] = np.where(

    regime_df[
        POA
    ] <= 100,

    "Low POA <= 100",

    "Active POA > 100"
)


regime_summary = (
    regime_df
    .groupby(
        "regime"
    )[TARGET]
    .agg(
        [
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max",
        ]
    )
    .reset_index()
)


regime_summary[
    "percentage"
] = (
        regime_summary[
            "count"
        ]
        /
        regime_summary[
            "count"
        ].sum()
        * 100
)


regime_summary.to_csv(
    EDA_DIR
    / "16_poa_operating_regime_summary.csv",
    index=False
)


print(
    "\nPOA operating regimes:"
)

print(
    regime_summary.to_string(
        index=False
    )
)


# ----------------------------------------------------------------
# 23. FINAL EDA INVENTORY
# ----------------------------------------------------------------

output_files = sorted(
    [
        path.name
        for path in EDA_DIR.iterdir()
        if path.is_file()
    ]
)


inventory_df = pd.DataFrame({

    "output_file": output_files
})


inventory_df.to_csv(
    EDA_DIR
    / "17_deep_eda_output_inventory.csv",
    index=False
)


# ----------------------------------------------------------------
# 24. FINAL MESSAGE
# ----------------------------------------------------------------

print(
    "\n" + "=" * 70
)

print(
    "DEEP EDA COMPLETE"
)

print(
    "=" * 70
)

print(
    f"\nAll outputs saved to:\n{EDA_DIR}"
)

print(
    f"\nTotal output files: "
    f"{len(output_files)}"
)

print(
    "\nMajor analyses completed:"
)

print(
    "1. Hour × Month generation heatmap"
)

print(
    "2. AC autocorrelation"
)

print(
    "3. AC ramp-rate analysis"
)

print(
    "4. POA irradiance-bin response"
)

print(
    "5. Cloud-cover regimes"
)

print(
    "6. Temperature relationships"
)

print(
    "7. Daily / monthly / annual energy"
)

print(
    "8. AC capacity utilization"
)

print(
    "9. kWh gross vs AC consistency"
)

print(
    "10. Daylight vs night behavior"
)

print(
    "11. Sensor relationships"
)

print(
    "12. Original data-quality patterns"
)

print(
    "13. Missingness patterns"
)

print(
    "14. Target correlations"
)

print(
    "15. POA operating regimes"
)

print(
    "\n" + "=" * 70
)