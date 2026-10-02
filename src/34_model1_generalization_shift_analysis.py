# Step 70 — Generalization / Distribution Shift Analysis
# ================================================================
# STEP 70 - MODEL 1 V17: GENERALIZATION / DISTRIBUTION SHIFT
# ================================================================
# Purpose:
# Compare the training-era operating distribution (2014-2016)
# against the later test-era distribution (2017-2018).
#
# This is a DIAGNOSTIC step.
# No ML model is trained here.
# ================================================================

from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

from scipy.stats import ks_2samp


print("=" * 70)
print("STEP 70 - MODEL 1 V17: GENERALIZATION / DISTRIBUTION SHIFT")
print("=" * 70)


# ----------------------------------------------------------------
# 1. FIND ACTUAL PROJECT ROOT
# ----------------------------------------------------------------

current_path = Path(__file__).resolve()

PROJECT_ROOT = None

KNOWN_DATA_FILE = (
    "PVDAQ_1433_ML_Ready_Preprocessed_Hourly.csv"
)

for parent in current_path.parents:

    if (
            (parent / ".venv").exists()
            and
            (
                    parent
                    / "data"
                    / "processed"
                    / KNOWN_DATA_FILE
            ).exists()
    ):
        PROJECT_ROOT = parent
        break

    if (
            (parent / ".venv_tf").exists()
            and
            (
                    parent
                    / "data"
                    / "processed"
                    / KNOWN_DATA_FILE
            ).exists()
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
# 2. FILES
# ----------------------------------------------------------------

FORECAST_FILE = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "Model1_WeatherSolar_NextHour.csv"
)

HISTORY_FILE = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "PVDAQ_1433_ML_Ready_Preprocessed_Hourly.csv"
)

OUTPUT_FILE = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "Model1_V17_Distribution_Shift_Analysis.csv"
)


# ----------------------------------------------------------------
# 3. LOAD DATA
# ----------------------------------------------------------------

forecast_df = pd.read_csv(
    FORECAST_FILE,
    parse_dates=[
        "timestamp",
        "target_timestamp"
    ]
)

history_df = pd.read_csv(
    HISTORY_FILE,
    parse_dates=["timestamp"]
)

history_df = (
    history_df
    .sort_values("timestamp")
    .reset_index(drop=True)
)


print(
    f"\nForecast dataset: {forecast_df.shape}"
)

print(
    f"History dataset : {history_df.shape}"
)


# ----------------------------------------------------------------
# 4. BASE FEATURES
# ----------------------------------------------------------------

base_features = [

    "ambient_temp__5062",
    "module_temp__5063",
    "poa_irradiance__5061",

    "T2M",
    "RH2M",
    "WS10M",
    "WD10M",
    "PS",
    "PRECTOTCORR",

    "ALLSKY_SFC_SW_DWN",
    "CLOUD_AMT",

    "high_cloud_cover",
    "medium_cloud_cover",
    "low_cloud_cover",

    "wind_gust_10m",
    "snowfall",

    "wind_speed_900_mb",
    "wind_direction_900_mb",

    "solar_zenith_angle",
    "solar_azimuth_angle",

    "hour_sin",
    "hour_cos",

    "month_sin",
    "month_cos",

    "day_of_year_sin",
    "day_of_year_cos",
]


target_col = "target_ac_power"


# ----------------------------------------------------------------
# 5. CREATE V14/V15 PHYSICS FEATURES
# ----------------------------------------------------------------

print("\nCreating physics-informed features...")


# Solar elevation
history_df["solar_elevation_deg"] = (
        90.0
        - history_df["solar_zenith_angle"]
)


# Sun cosine of zenith
zenith_rad = np.deg2rad(
    history_df["solar_zenith_angle"]
)

history_df["sun_cos_zenith"] = np.maximum(
    np.cos(zenith_rad),
    0.0
)


# Daylight flag
history_df["daylight_flag"] = (
        history_df["solar_zenith_angle"] < 90
).astype(int)


# Relative air mass
zenith_for_airmass = (
    history_df["solar_zenith_angle"]
    .clip(
        lower=0,
        upper=89.9
    )
)

relative_airmass = (
    pvlib.atmosphere
    .get_relative_airmass(
        zenith_for_airmass,
        model="kastenyoung1989"
    )
)

relative_airmass = pd.Series(
    relative_airmass,
    index=history_df.index
)

relative_airmass.loc[
    history_df["daylight_flag"] == 0
    ] = 0

history_df["relative_airmass"] = (
    relative_airmass
)


# Pressure corrected air mass
history_df[
    "pressure_corrected_airmass"
] = (
        history_df["relative_airmass"]
        * history_df["PS"]
        / 1013.25
)


# Thermal difference
history_df[
    "module_ambient_delta"
] = (
        history_df["module_temp__5063"]
        - history_df["ambient_temp__5062"]
)


# POA × module temperature
history_df[
    "poa_module_temp_interaction"
] = (
        history_df["poa_irradiance__5061"]
        * history_df["module_temp__5063"]
)


# POA × ambient temperature
history_df[
    "poa_ambient_temp_interaction"
] = (
        history_df["poa_irradiance__5061"]
        * history_df["ambient_temp__5062"]
)


physics_features = [

    "solar_elevation_deg",
    "sun_cos_zenith",
    "daylight_flag",
    "relative_airmass",
    "pressure_corrected_airmass",
    "module_ambient_delta",
    "poa_module_temp_interaction",
    "poa_ambient_temp_interaction",
]


all_features = (
        base_features
        + physics_features
)


print(
    f"Total features analyzed: "
    f"{len(all_features)}"
)


# ----------------------------------------------------------------
# 6. MERGE PHYSICS FEATURES
# ----------------------------------------------------------------

physics_source = history_df[
    ["timestamp"] + physics_features
    ].copy()


forecast_df = forecast_df.merge(
    physics_source,
    on="timestamp",
    how="left",
    validate="one_to_one"
)


# ----------------------------------------------------------------
# 7. DEFINE TARGET TIMESTAMP
# ----------------------------------------------------------------

forecast_df["target_year"] = (
    forecast_df["target_timestamp"].dt.year
)


# ----------------------------------------------------------------
# 8. DEFINE PERIODS
# ----------------------------------------------------------------
# Training-era distribution:
#   2014-2016
#
# Later test-era distribution:
#   2017-2018

train_period = forecast_df[
    (
            forecast_df["target_timestamp"]
            >= pd.Timestamp("2014-01-01")
    )
    &
    (
            forecast_df["target_timestamp"]
            <= pd.Timestamp("2016-12-31 23:59:59")
    )
    ].copy()


test_period = forecast_df[
    (
            forecast_df["target_timestamp"]
            >= pd.Timestamp("2017-01-01")
    )
    &
    (
            forecast_df["target_timestamp"]
            <= pd.Timestamp("2018-12-31 23:59:59")
    )
    ].copy()


print("\n" + "-" * 70)
print("PERIODS")
print("-" * 70)

print(
    f"2014-2016 rows: {len(train_period)}"
)

print(
    f"2017-2018 rows: {len(test_period)}"
)


# ----------------------------------------------------------------
# 9. TARGET / OPERATING REGIME SUMMARY
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("AC POWER DISTRIBUTION SHIFT")
print("=" * 70)


def print_target_summary(name, data):

    y = data[target_col].dropna()

    active_mask = y > 1.0

    print(
        f"\n{name}"
    )

    print(
        f"Rows              : {len(y)}"
    )

    print(
        f"Mean AC power     : {y.mean():.4f}"
    )

    print(
        f"Median AC power   : {y.median():.4f}"
    )

    print(
        f"Std AC power      : {y.std():.4f}"
    )

    print(
        f"Min               : {y.min():.4f}"
    )

    print(
        f"Max               : {y.max():.4f}"
    )

    print(
        f"Active >1 kW      : "
        f"{active_mask.mean() * 100:.2f}%"
    )

    print(
        f"Near-zero <=1 kW  : "
        f"{(~active_mask).mean() * 100:.2f}%"
    )


print_target_summary(
    "TRAINING ERA (2014-2016)",
    train_period
)

print_target_summary(
    "TEST ERA (2017-2018)",
    test_period
)


# ----------------------------------------------------------------
# 10. YEAR-WISE TARGET SUMMARY
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("YEAR-WISE AC POWER SUMMARY")
print("=" * 70)


year_rows = []


for year in sorted(
        forecast_df["target_year"]
                .dropna()
                .unique()
):

    year_data = forecast_df[
        forecast_df["target_year"] == year
        ]

    y = year_data[
        target_col
    ].dropna()

    if len(y) == 0:
        continue

    year_rows.append({

        "year": int(year),

        "rows": len(y),

        "mean_ac_power": y.mean(),

        "median_ac_power": y.median(),

        "std_ac_power": y.std(),

        "max_ac_power": y.max(),

        "active_percentage":
            (y > 1.0).mean() * 100,
    })


year_summary = pd.DataFrame(
    year_rows
)


print(
    year_summary.to_string(
        index=False
    )
)


# ----------------------------------------------------------------
# 11. FEATURE DISTRIBUTION COMPARISON
# ----------------------------------------------------------------
# Metrics:
#
# mean_train
# mean_test
# std_train
# std_test
# median_train
# median_test
# standardized_mean_difference
# relative_mean_change_pct
# KS statistic
# KS p-value
#
# Standardized mean difference:
#
# (test_mean - train_mean)
# --------------------------------
# pooled standard deviation
#
# Large absolute values indicate stronger shift.

print("\n" + "=" * 70)
print("FEATURE DISTRIBUTION SHIFT")
print("=" * 70)


results = []


for feature in all_features:

    train_values = (
        train_period[feature]
        .dropna()
        .to_numpy()
    )

    test_values = (
        test_period[feature]
        .dropna()
        .to_numpy()
    )


    if (
            len(train_values) == 0
            or
            len(test_values) == 0
    ):
        continue


    train_mean = np.mean(
        train_values
    )

    test_mean = np.mean(
        test_values
    )


    train_std = np.std(
        train_values,
        ddof=1
    )

    test_std = np.std(
        test_values,
        ddof=1
    )


    train_median = np.median(
        train_values
    )

    test_median = np.median(
        test_values
    )


    # ------------------------------------------------------------
    # Pooled standard deviation
    # ------------------------------------------------------------

    n_train = len(train_values)
    n_test = len(test_values)

    pooled_std = np.sqrt(

        (
                (n_train - 1)
                * train_std ** 2

                +

                (n_test - 1)
                * test_std ** 2
        )
        /
        (
                n_train
                + n_test
                - 2
        )
    )


    if pooled_std > 0:

        standardized_mean_difference = (
                (test_mean - train_mean)
                / pooled_std
        )

    else:

        standardized_mean_difference = 0.0


    # ------------------------------------------------------------
    # Relative mean change
    # ------------------------------------------------------------

    if abs(train_mean) > 1e-12:

        relative_mean_change_pct = (
                (test_mean - train_mean)
                / abs(train_mean)
                * 100
        )

    else:

        relative_mean_change_pct = np.nan


    # ------------------------------------------------------------
    # Kolmogorov-Smirnov test
    # ------------------------------------------------------------

    ks_statistic, ks_pvalue = (
        ks_2samp(
            train_values,
            test_values
        )
    )


    results.append({

        "feature": feature,

        "train_mean": train_mean,

        "test_mean": test_mean,

        "train_std": train_std,

        "test_std": test_std,

        "train_median": train_median,

        "test_median": test_median,

        "standardized_mean_difference":
            standardized_mean_difference,

        "relative_mean_change_pct":
            relative_mean_change_pct,

        "ks_statistic":
            ks_statistic,

        "ks_pvalue":
            ks_pvalue,

    })


shift_df = pd.DataFrame(
    results
)


# ----------------------------------------------------------------
# 12. ABSOLUTE SHIFT SCORE
# ----------------------------------------------------------------

shift_df[
    "abs_standardized_mean_difference"
] = shift_df[
    "standardized_mean_difference"
].abs()


shift_df[
    "abs_relative_mean_change_pct"
] = shift_df[
    "relative_mean_change_pct"
].abs()


shift_df = shift_df.sort_values(
    "abs_standardized_mean_difference",
    ascending=False
).reset_index(drop=True)


# ----------------------------------------------------------------
# 13. PRINT TOP SHIFTED FEATURES
# ----------------------------------------------------------------

print(
    "\nTop features by absolute standardized "
    "mean difference:"
)

print(
    shift_df[
        [
            "feature",
            "train_mean",
            "test_mean",
            "standardized_mean_difference",
            "relative_mean_change_pct",
            "ks_statistic",
            "ks_pvalue",
        ]
    ]
    .head(15)
    .to_string(index=False)
)


# ----------------------------------------------------------------
# 14. FLAG STRONG SHIFTS
# ----------------------------------------------------------------
#
# These are diagnostic flags, NOT hard scientific laws.
#
# SMD >= 0.50 = notable distribution difference
#
# KS p-value < 0.05 = statistically detectable difference
#
# With large datasets, KS p-values can become very small even
# for practically small differences. Therefore we look at
# effect size + KS together.

shift_df["notable_smd"] = (
        shift_df[
            "abs_standardized_mean_difference"
        ] >= 0.50
)


shift_df["ks_detectable"] = (
        shift_df["ks_pvalue"] < 0.05
)


strong_shift_count = (
    shift_df["notable_smd"]
    .sum()
)


print("\n" + "-" * 70)

print(
    f"Features with |SMD| >= 0.50: "
    f"{strong_shift_count}"
)

print(
    f"Features with KS p-value < 0.05: "
    f"{shift_df['ks_detectable'].sum()}"
)


# ----------------------------------------------------------------
# 15. CATEGORY SUMMARY
# ----------------------------------------------------------------

categories = {

    "PV / Irradiance": [
        "poa_irradiance__5061",
        "ALLSKY_SFC_SW_DWN",
        "CLOUD_AMT",
    ],

    "Temperature": [
        "ambient_temp__5062",
        "module_temp__5063",
        "T2M",
        "module_ambient_delta",
    ],

    "Cloud": [
        "high_cloud_cover",
        "medium_cloud_cover",
        "low_cloud_cover",
    ],

    "Wind": [
        "WS10M",
        "WD10M",
        "wind_gust_10m",
        "wind_speed_900_mb",
        "wind_direction_900_mb",
    ],

    "Atmosphere": [
        "RH2M",
        "PS",
        "PRECTOTCORR",
        "snowfall",
        "relative_airmass",
        "pressure_corrected_airmass",
    ],

    "Solar Geometry": [
        "solar_zenith_angle",
        "solar_azimuth_angle",
        "solar_elevation_deg",
        "sun_cos_zenith",
        "daylight_flag",
    ],

    "Interactions": [
        "poa_module_temp_interaction",
        "poa_ambient_temp_interaction",
    ],
}


print("\n" + "=" * 70)
print("CATEGORY-LEVEL SHIFT")
print("=" * 70)


category_rows = []


for category, feature_list in categories.items():

    available = shift_df[
        shift_df["feature"].isin(feature_list)
    ]

    if len(available) == 0:
        continue


    category_rows.append({

        "category": category,

        "feature_count":
            len(available),

        "mean_abs_smd":
            available[
                "abs_standardized_mean_difference"
            ].mean(),

        "max_abs_smd":
            available[
                "abs_standardized_mean_difference"
            ].max(),

        "mean_abs_relative_change_pct":
            available[
                "abs_relative_mean_change_pct"
            ].mean(),

    })


category_df = pd.DataFrame(
    category_rows
).sort_values(
    "mean_abs_smd",
    ascending=False
)


print(
    category_df.to_string(
        index=False
    )
)


# ----------------------------------------------------------------
# 16. FEATURE-SPECIFIC YEARLY MEAN CHECK
# ----------------------------------------------------------------

selected_features = [

    "poa_irradiance__5061",
    "ambient_temp__5062",
    "module_temp__5063",
    "ALLSKY_SFC_SW_DWN",
    "CLOUD_AMT",
    "WS10M",
    "RH2M",
    "AC_POWER_PLACEHOLDER",
]


print("\n" + "=" * 70)
print("SELECTED FEATURE YEARLY MEANS")
print("=" * 70)


year_feature_rows = []


real_selected_features = [
    f
    for f in selected_features
    if f != "AC_POWER_PLACEHOLDER"
]


for year in sorted(
        forecast_df["target_year"]
                .dropna()
                .unique()
):

    year_data = forecast_df[
        forecast_df["target_year"] == year
        ]


    row = {
        "year": int(year)
    }


    for feature in real_selected_features:

        row[
            feature
        ] = year_data[
            feature
        ].mean()


    row[
        "mean_ac_power"
    ] = year_data[
        target_col
    ].mean()


    year_feature_rows.append(
        row
    )


year_feature_df = pd.DataFrame(
    year_feature_rows
)


print(
    year_feature_df.to_string(
        index=False
    )
)


# ----------------------------------------------------------------
# 17. SAVE MAIN SHIFT TABLE
# ----------------------------------------------------------------

shift_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ----------------------------------------------------------------
# 18. SAVE CATEGORY TABLE
# ----------------------------------------------------------------

CATEGORY_OUTPUT_FILE = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "Model1_V17_Category_Shift.csv"
)


category_df.to_csv(
    CATEGORY_OUTPUT_FILE,
    index=False
)


# ----------------------------------------------------------------
# 19. SAVE YEAR SUMMARY
# ----------------------------------------------------------------

YEAR_OUTPUT_FILE = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "Model1_V17_Yearly_Shift_Summary.csv"
)


year_summary.to_csv(
    YEAR_OUTPUT_FILE,
    index=False
)


# ----------------------------------------------------------------
# 20. FINAL MESSAGE
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("V17 DISTRIBUTION SHIFT ANALYSIS COMPLETE")
print("=" * 70)

print(
    "\nMain feature-shift file:"
)

print(
    OUTPUT_FILE
)

print(
    "\nCategory-shift file:"
)

print(
    CATEGORY_OUTPUT_FILE
)

print(
    "\nYearly summary file:"
)

print(
    YEAR_OUTPUT_FILE
)

print(
    "\nNext step: use the actual shift evidence to design V18."
)

print("=" * 70)