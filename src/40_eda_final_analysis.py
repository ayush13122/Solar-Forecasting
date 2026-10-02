# Step 76 — Final EDA
# ================================================================
# STEP 76 - FINAL EDA
# ================================================================
# Formal EDA for the cleaned hourly ML-ready dataset.
#
# No model training
# No test-set evaluation
# No modification of source dataset
# ================================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


print("=" * 70)
print("STEP 76 - FINAL EDA")
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
# 2. PATHS
# ----------------------------------------------------------------

INPUT_FILE = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / KNOWN_DATA_FILE
)

EDA_DIR = (
        PROJECT_ROOT
        / "reports"
        / "eda"
)

EDA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ----------------------------------------------------------------
# 3. LOAD DATA
# ----------------------------------------------------------------

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["timestamp"]
)

df = (
    df
    .sort_values("timestamp")
    .reset_index(drop=True)
)


print(
    f"\nDataset shape: {df.shape}"
)

print(
    f"Date range: "
    f"{df['timestamp'].min()} → "
    f"{df['timestamp'].max()}"
)


# ----------------------------------------------------------------
# 4. BASIC STRUCTURE
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("BASIC DATASET STRUCTURE")
print("=" * 70)

print(
    f"Rows    : {df.shape[0]}"
)

print(
    f"Columns : {df.shape[1]}"
)

print("\nColumns:")

for col in df.columns:
    print(
        f"- {col}"
    )


# ----------------------------------------------------------------
# 5. DESCRIPTIVE STATISTICS
# ----------------------------------------------------------------

numeric_df = df.select_dtypes(
    include=np.number
)

descriptive_stats = (
    numeric_df
    .describe()
    .T
)

descriptive_stats.to_csv(
    EDA_DIR
    / "EDA_descriptive_statistics.csv"
)


print("\nDescriptive statistics saved.")


# ----------------------------------------------------------------
# 6. TARGET IDENTIFICATION
# ----------------------------------------------------------------

target_candidates = [
    "ac_power__5069",
    "target_ac_power"
]

target_col = None

for candidate in target_candidates:

    if candidate in df.columns:
        target_col = candidate
        break


if target_col is None:

    raise ValueError(
        "AC power target column not found."
    )


print(
    f"\nTarget column: {target_col}"
)


# ----------------------------------------------------------------
# 7. TARGET SUMMARY
# ----------------------------------------------------------------

y = df[target_col].dropna()

target_summary = pd.DataFrame({

    "metric": [

        "count",
        "mean",
        "median",
        "std",
        "min",
        "max",
        "zero_count",
        "zero_percentage",
        "positive_count",
        "positive_percentage",
    ],

    "value": [

        len(y),

        y.mean(),

        y.median(),

        y.std(),

        y.min(),

        y.max(),

        (y <= 1).sum(),

        (y <= 1).mean() * 100,

        (y > 1).sum(),

        (y > 1).mean() * 100,
        ]
})


target_summary.to_csv(
    EDA_DIR
    / "EDA_target_summary.csv",
    index=False
)


print(
    "\nTarget summary:"
)

print(
    target_summary.to_string(
        index=False
    )
)


# ----------------------------------------------------------------
# 8. TIME FEATURES FOR EDA
# ----------------------------------------------------------------

df["year"] = (
    df["timestamp"].dt.year
)

df["month"] = (
    df["timestamp"].dt.month
)

df["hour"] = (
    df["timestamp"].dt.hour
)

df["day_of_year"] = (
    df["timestamp"].dt.dayofyear
)


# ----------------------------------------------------------------
# 9. DAYLIGHT FLAG
# ----------------------------------------------------------------

if "solar_zenith_angle" in df.columns:

    df["eda_daylight"] = (
            df["solar_zenith_angle"] < 90
    ).astype(int)

else:

    df["eda_daylight"] = (
            df["poa_irradiance__5061"] > 0
    ).astype(int)


# ----------------------------------------------------------------
# 10. AC DISTRIBUTION
# ----------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.hist(
    y,
    bins=60
)

plt.xlabel(
    "AC Power (kW)"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "AC Power Distribution"
)

plt.tight_layout()

plt.savefig(
    EDA_DIR
    / "01_ac_power_distribution.png",
    dpi=200
)

plt.close()


# ----------------------------------------------------------------
# 11. POA DISTRIBUTION
# ----------------------------------------------------------------

if "poa_irradiance__5061" in df.columns:

    poa = (
        df[
            "poa_irradiance__5061"
        ]
        .dropna()
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        poa,
        bins=60
    )

    plt.xlabel(
        "POA Irradiance"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "POA Irradiance Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        EDA_DIR
        / "02_poa_distribution.png",
        dpi=200
    )

    plt.close()


# ----------------------------------------------------------------
# 12. POA VS AC POWER
# ----------------------------------------------------------------

if "poa_irradiance__5061" in df.columns:

    scatter_df = df[
        [
            "poa_irradiance__5061",
            target_col
        ]
    ].dropna()

    if len(scatter_df) > 10000:

        scatter_df = scatter_df.sample(
            10000,
            random_state=42
        )

    plt.figure(
        figsize=(10, 6)
    )

    plt.scatter(
        scatter_df[
            "poa_irradiance__5061"
        ],
        scatter_df[
            target_col
        ],
        s=8,
        alpha=0.25
    )

    plt.xlabel(
        "POA Irradiance"
    )

    plt.ylabel(
        "AC Power (kW)"
    )

    plt.title(
        "POA Irradiance vs AC Power"
    )

    plt.tight_layout()

    plt.savefig(
        EDA_DIR
        / "03_poa_vs_ac_power.png",
        dpi=200
    )

    plt.close()


# ----------------------------------------------------------------
# 13. HOURLY AC PROFILE
# ----------------------------------------------------------------

hourly_profile = (
    df.groupby("hour")[
        target_col
    ]
    .agg(
        [
            "mean",
            "median",
            "std"
        ]
    )
    .reset_index()
)


hourly_profile.to_csv(
    EDA_DIR
    / "EDA_hourly_ac_profile.csv",
    index=False
)


plt.figure(
    figsize=(10, 6)
)

plt.plot(
    hourly_profile["hour"],
    hourly_profile["mean"]
)

plt.xlabel(
    "Hour of Day"
)

plt.ylabel(
    "Mean AC Power (kW)"
)

plt.title(
    "Average AC Power by Hour"
)

plt.xticks(
    range(0, 24)
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    EDA_DIR
    / "04_hourly_ac_profile.png",
    dpi=200
)

plt.close()


# ----------------------------------------------------------------
# 14. MONTHLY AC PROFILE
# ----------------------------------------------------------------

monthly_profile = (
    df.groupby("month")[
        target_col
    ]
    .agg(
        [
            "mean",
            "median",
            "std"
        ]
    )
    .reset_index()
)


monthly_profile.to_csv(
    EDA_DIR
    / "EDA_monthly_ac_profile.csv",
    index=False
)


plt.figure(
    figsize=(10, 6)
)

plt.plot(
    monthly_profile["month"],
    monthly_profile["mean"],
    marker="o"
)

plt.xlabel(
    "Month"
)

plt.ylabel(
    "Mean AC Power (kW)"
)

plt.title(
    "Average AC Power by Month"
)

plt.xticks(
    range(1, 13)
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    EDA_DIR
    / "05_monthly_ac_profile.png",
    dpi=200
)

plt.close()


# ----------------------------------------------------------------
# 15. ANNUAL AC PROFILE
# ----------------------------------------------------------------

annual_profile = (
    df.groupby("year")[
        target_col
    ]
    .agg(
        [
            "mean",
            "median",
            "std",
            "max"
        ]
    )
    .reset_index()
)


annual_profile.to_csv(
    EDA_DIR
    / "EDA_annual_ac_profile.csv",
    index=False
)


plt.figure(
    figsize=(10, 6)
)

plt.plot(
    annual_profile["year"],
    annual_profile["mean"],
    marker="o"
)

plt.xlabel(
    "Year"
)

plt.ylabel(
    "Mean AC Power (kW)"
)

plt.title(
    "Average AC Power by Year"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    EDA_DIR
    / "06_annual_ac_profile.png",
    dpi=200
)

plt.close()


# ----------------------------------------------------------------
# 16. AMBIENT VS MODULE TEMPERATURE
# ----------------------------------------------------------------

temperature_cols = [
    "ambient_temp__5062",
    "module_temp__5063"
]

available_temp_cols = [
    col
    for col in temperature_cols
    if col in df.columns
]


if len(available_temp_cols) == 2:

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        df["timestamp"],
        df["ambient_temp__5062"],
        linewidth=0.7,
        label="Ambient"
    )

    plt.plot(
        df["timestamp"],
        df["module_temp__5063"],
        linewidth=0.7,
        label="Module"
    )

    plt.xlabel(
        "Time"
    )

    plt.ylabel(
        "Temperature (°C)"
    )

    plt.title(
        "Ambient vs Module Temperature"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        EDA_DIR
        / "07_ambient_vs_module_temperature.png",
        dpi=200
    )

    plt.close()


# ----------------------------------------------------------------
# 17. AC POWER BY DAYLIGHT
# ----------------------------------------------------------------

daylight_summary = (
    df.groupby("eda_daylight")[
        target_col
    ]
    .agg(
        [
            "count",
            "mean",
            "median",
            "std",
            "max"
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
        1: "Daylight"
    }
)


daylight_summary.to_csv(
    EDA_DIR
    / "EDA_daylight_summary.csv",
    index=False
)


# ----------------------------------------------------------------
# 18. CORRELATION ANALYSIS
# ----------------------------------------------------------------

correlation_features = [

    "ac_power__5069",

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

    "solar_zenith_angle",

    "solar_azimuth_angle",
]


available_corr_features = [
    col
    for col in correlation_features
    if col in df.columns
]


corr_matrix = (
    df[
        available_corr_features
    ]
    .corr()
)


corr_matrix.to_csv(
    EDA_DIR
    / "EDA_correlation_matrix.csv"
)


# ----------------------------------------------------------------
# 19. CORRELATION WITH AC POWER
# ----------------------------------------------------------------

if target_col in corr_matrix.columns:

    target_correlations = (
        corr_matrix[
            target_col
        ]
        .sort_values(
            ascending=False
        )
        .to_frame(
            name="correlation_with_ac_power"
        )
    )


    target_correlations.to_csv(
        EDA_DIR
        / "EDA_target_correlations.csv"
    )


    print(
        "\nTop correlations with AC power:"
    )

    print(
        target_correlations.head(
            10
        ).to_string()
    )


# ----------------------------------------------------------------
# 20. EDA SUMMARY TABLE
# ----------------------------------------------------------------

eda_summary = pd.DataFrame({

    "item": [

        "rows",

        "columns",

        "start_timestamp",

        "end_timestamp",

        "duplicate_timestamps",

        "missing_values_total",

        "target_mean",

        "target_median",

        "target_std",

        "target_min",

        "target_max",

        "target_zero_or_near_zero_pct",

        "target_active_pct",
    ],

    "value": [

        len(df),

        len(df.columns),

        str(df["timestamp"].min()),

        str(df["timestamp"].max()),

        int(
            df["timestamp"]
            .duplicated()
            .sum()
        ),

        int(
            df.isna()
            .sum()
            .sum()
        ),

        y.mean(),

        y.median(),

        y.std(),

        y.min(),

        y.max(),

        (y <= 1).mean() * 100,

        (y > 1).mean() * 100,
        ]
})


eda_summary.to_csv(
    EDA_DIR
    / "EDA_final_summary.csv",
    index=False
)


# ----------------------------------------------------------------
# 21. FINAL MESSAGE
# ----------------------------------------------------------------

print(
    "\n" + "=" * 70
)

print(
    "EDA COMPLETE"
)

print(
    "=" * 70
)

print(
    f"\nEDA outputs saved to:\n{EDA_DIR}"
)

print(
    "\nGenerated:"
)

print(
    "• descriptive statistics"
)

print(
    "• target summary"
)

print(
    "• hourly profile"
)

print(
    "• monthly profile"
)

print(
    "• annual profile"
)

print(
    "• daylight summary"
)

print(
    "• correlation analysis"
)

print(
    "• AC / POA distributions"
)

print(
    "• POA vs AC relationship"
)

print(
    "• temperature analysis"
)

print(
    "\n" + "=" * 70
)