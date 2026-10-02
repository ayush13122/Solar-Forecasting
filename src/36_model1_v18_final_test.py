# V18 final confirmatory test karte hain.

# ================================================================
# STEP 72 - MODEL 1 V18 FINAL TEST
# ================================================================
# Selected V18 configuration:
#   Recency half-life = 365 days
#
# Train:
#   2014-2016
#
# Test:
#   2017-2018
#
# Architecture:
#   Stage 1 -> Active / Near-zero classifier
#   Stage 2 -> Active-power regressor
#
# Recency weighting:
#   weight = 0.5 ** (age_days / 365)
#
# Best iterations are obtained automatically from the
# V18 rolling-validation results.
# ================================================================

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import pvlib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


print("=" * 70)
print("STEP 72 - MODEL 1 V18 FINAL TEST")
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
# 2. FILE PATHS
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
        / KNOWN_DATA_FILE
)

VALIDATION_FILE = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "Model1_V18_Recency_Weighted_Validation.csv"
)

PREDICTION_FILE = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "Model1_V18_Final_Test_Predictions.csv"
)

RESULT_FILE = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "Model1_V18_Final_Test_Results.csv"
)


# ----------------------------------------------------------------
# 3. SETTINGS
# ----------------------------------------------------------------

HALF_LIFE_DAYS = 365

ACTIVE_THRESHOLD = 1.0

CLASSIFICATION_THRESHOLD = 0.50


# ----------------------------------------------------------------
# 4. LOAD DATA
# ----------------------------------------------------------------

forecast_df = pd.read_csv(
    FORECAST_FILE,
    parse_dates=[
        "timestamp",
        "target_timestamp",
    ],
)

history_df = pd.read_csv(
    HISTORY_FILE,
    parse_dates=[
        "timestamp",
    ],
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
# 5. LOAD V18 VALIDATION RESULTS
# ----------------------------------------------------------------

validation_df = pd.read_csv(
    VALIDATION_FILE
)


selected_validation = validation_df[
    validation_df[
        "half_life_days"
    ] == HALF_LIFE_DAYS
    ].copy()


if len(selected_validation) == 0:
    raise ValueError(
        "No V18 validation results found "
        f"for half-life {HALF_LIFE_DAYS}."
    )


classifier_estimators = int(
    round(
        selected_validation[
            "classifier_best_iteration"
        ].mean()
    )
)


regressor_estimators = int(
    round(
        selected_validation[
            "regressor_best_iteration"
        ].mean()
    )
)


print("\nV18 selected configuration:")

print(
    f"Half-life: "
    f"{HALF_LIFE_DAYS} days"
)

print(
    f"Classifier estimators: "
    f"{classifier_estimators}"
)

print(
    f"Regressor estimators: "
    f"{regressor_estimators}"
)


# ----------------------------------------------------------------
# 6. BASE FEATURES
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
# 7. CREATE PHYSICS FEATURES
# ----------------------------------------------------------------

print(
    "\nCreating physics-informed features..."
)


# Solar elevation
history_df[
    "solar_elevation_deg"
] = (
        90.0
        - history_df["solar_zenith_angle"]
)


# Sun cosine of zenith
zenith_rad = np.deg2rad(
    history_df["solar_zenith_angle"]
)

history_df[
    "sun_cos_zenith"
] = np.maximum(
    np.cos(zenith_rad),
    0.0
)


# Daylight flag
history_df[
    "daylight_flag"
] = (
        history_df["solar_zenith_angle"] < 90
).astype(int)


# Relative air mass
zenith_for_airmass = (
    history_df["solar_zenith_angle"]
    .clip(
        lower=0,
        upper=89.9,
    )
)


relative_airmass = (
    pvlib.atmosphere
    .get_relative_airmass(
        zenith_for_airmass,
        model="kastenyoung1989",
    )
)


relative_airmass = pd.Series(
    relative_airmass,
    index=history_df.index,
)


relative_airmass.loc[
    history_df["daylight_flag"] == 0
    ] = 0


history_df[
    "relative_airmass"
] = relative_airmass


# Pressure-corrected air mass
history_df[
    "pressure_corrected_airmass"
] = (
        history_df["relative_airmass"]
        * history_df["PS"]
        / 1013.25
)


# Module - ambient temperature
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


v18_features = (
        base_features
        + physics_features
)


print(
    f"\nBase features     : "
    f"{len(base_features)}"
)

print(
    f"Physics features  : "
    f"{len(physics_features)}"
)

print(
    f"Total V18 features: "
    f"{len(v18_features)}"
)


# ----------------------------------------------------------------
# 8. MERGE PHYSICS FEATURES
# ----------------------------------------------------------------

physics_source = history_df[
    ["timestamp"] + physics_features
    ].copy()


forecast_df = forecast_df.merge(
    physics_source,
    on="timestamp",
    how="left",
    validate="one_to_one",
)


# ----------------------------------------------------------------
# 9. MERGE CURRENT AC FOR PERSISTENCE
# ----------------------------------------------------------------

current_ac_source = history_df[
    [
        "timestamp",
        "ac_power__5069",
    ]
].copy()


current_ac_source = current_ac_source.rename(
    columns={
        "ac_power__5069":
            "current_ac_power"
    }
)


forecast_df = forecast_df.merge(
    current_ac_source,
    on="timestamp",
    how="left",
    validate="one_to_one",
)


# ----------------------------------------------------------------
# 10. MISSING CHECK
# ----------------------------------------------------------------

missing_features = (
    forecast_df[v18_features]
    .isna()
    .sum()
)


print(
    "\nMissing V18 feature values:"
)

print(
    missing_features[
        missing_features > 0
        ]
)


if missing_features.sum() > 0:

    raise ValueError(
        "Missing V18 feature values detected."
    )


# ----------------------------------------------------------------
# 11. CREATE ACTIVE TARGET
# ----------------------------------------------------------------

forecast_df[
    "active_target"
] = (
        forecast_df[target_col]
        > ACTIVE_THRESHOLD
).astype(int)


# ----------------------------------------------------------------
# 12. TRAIN / TEST SPLIT
# ----------------------------------------------------------------

train_mask = (

        (
                forecast_df["target_timestamp"]
                >= pd.Timestamp("2014-01-01")
        )

        &

        (
                forecast_df["target_timestamp"]
                <= pd.Timestamp(
            "2016-12-31 23:59:59"
        )
        )

        &

        forecast_df[target_col].notna()
)


test_mask = (

        (
                forecast_df["target_timestamp"]
                >= pd.Timestamp("2017-01-01")
        )

        &

        (
                forecast_df["target_timestamp"]
                <= pd.Timestamp(
            "2018-12-31 23:59:59"
        )
        )

        &

        forecast_df[target_col].notna()
)


train_data = forecast_df.loc[
    train_mask
].copy()


test_data = forecast_df.loc[
    test_mask
].copy()


print(
    "\n" + "-" * 70
)

print(
    "FINAL TRAIN / TEST SPLIT"
)

print(
    "-" * 70
)

print(
    f"Train rows: "
    f"{len(train_data)}"
)

print(
    f"Test rows : "
    f"{len(test_data)}"
)


# ----------------------------------------------------------------
# 13. CALCULATE RECENCY WEIGHTS
# ----------------------------------------------------------------

reference_time = (
    train_data[
        "target_timestamp"
    ].max()
)


age_days = (
        (
                reference_time
                - train_data[
                    "target_timestamp"
                ]
        )
        .dt.total_seconds()
        / 86400.0
)


train_data[
    "recency_weight"
] = (
        0.5
        ** (
                age_days
                / HALF_LIFE_DAYS
        )
)


# Normalize mean weight = 1
train_data[
    "recency_weight"
] = (
        train_data[
            "recency_weight"
        ]
        /
        train_data[
            "recency_weight"
        ].mean()
)


print(
    "\nRecency weights:"
)

print(
    f"Min : "
    f"{train_data['recency_weight'].min():.4f}"
)

print(
    f"Max : "
    f"{train_data['recency_weight'].max():.4f}"
)

print(
    f"Mean: "
    f"{train_data['recency_weight'].mean():.4f}"
)


# ----------------------------------------------------------------
# 14. STAGE 1 - CLASSIFIER
# ----------------------------------------------------------------

print(
    "\n" + "=" * 70
)

print(
    "STAGE 1 - ACTIVE / NEAR-ZERO CLASSIFIER"
)

print(
    "=" * 70
)


X_train_cls = train_data[
    v18_features
]

y_train_cls = train_data[
    "active_target"
]


X_test_cls = test_data[
    v18_features
]

y_test_cls = test_data[
    "active_target"
]


classifier = lgb.LGBMClassifier(

    objective="binary",

    n_estimators=
    classifier_estimators,

    learning_rate=0.05,

    num_leaves=31,

    max_depth=-1,

    subsample=0.9,

    colsample_bytree=0.9,

    random_state=42,

    n_jobs=-1,
)


classifier.fit(

    X_train_cls,

    y_train_cls,

    sample_weight=
    train_data[
        "recency_weight"
    ],
)


test_probability = (
    classifier
    .predict_proba(
        X_test_cls
    )[:, 1]
)


test_active_prediction = (
        test_probability
        >= CLASSIFICATION_THRESHOLD
).astype(int)


classification_accuracy = (
    accuracy_score(
        y_test_cls,
        test_active_prediction,
    )
)


classification_precision = (
    precision_score(
        y_test_cls,
        test_active_prediction,
        zero_division=0,
    )
)


classification_recall = (
    recall_score(
        y_test_cls,
        test_active_prediction,
        zero_division=0,
    )
)


classification_f1 = (
    f1_score(
        y_test_cls,
        test_active_prediction,
        zero_division=0,
    )
)


print(
    f"\nAccuracy : "
    f"{classification_accuracy:.4f}"
)

print(
    f"Precision: "
    f"{classification_precision:.4f}"
)

print(
    f"Recall   : "
    f"{classification_recall:.4f}"
)

print(
    f"F1       : "
    f"{classification_f1:.4f}"
)


# ----------------------------------------------------------------
# 15. STAGE 2 - ACTIVE POWER REGRESSOR
# ----------------------------------------------------------------

print(
    "\n" + "=" * 70
)

print(
    "STAGE 2 - ACTIVE POWER REGRESSOR"
)

print(
    "=" * 70
)


active_train = train_data[
    train_data[
        "active_target"
    ] == 1
    ].copy()


X_train_reg = active_train[
    v18_features
]

y_train_reg = active_train[
    target_col
]


print(
    f"\nActive training rows: "
    f"{len(active_train)}"
)


# Re-normalize weights within active subset
active_weights = (
        active_train[
            "recency_weight"
        ]
        /
        active_train[
            "recency_weight"
        ].mean()
)


regressor = lgb.LGBMRegressor(

    objective="regression",

    n_estimators=
    regressor_estimators,

    learning_rate=0.05,

    num_leaves=31,

    max_depth=-1,

    subsample=0.9,

    colsample_bytree=0.9,

    random_state=42,

    n_jobs=-1,
)


regressor.fit(

    X_train_reg,

    y_train_reg,

    sample_weight=active_weights,
)


# Stage-2 prediction for all test rows
regression_prediction = (
    regressor.predict(
        X_test_cls
    )
)


# No negative generation
regression_prediction = np.maximum(
    regression_prediction,
    0.0,
)


# ----------------------------------------------------------------
# 16. FINAL TWO-STAGE PREDICTION
# ----------------------------------------------------------------

final_prediction = np.where(

    test_active_prediction == 1,

    regression_prediction,

    0.0,
    )


final_prediction = np.maximum(
    final_prediction,
    0.0,
)


# ----------------------------------------------------------------
# 17. FINAL TEST METRICS
# ----------------------------------------------------------------

y_test = test_data[
    target_col
].to_numpy()


final_mae = (
    mean_absolute_error(
        y_test,
        final_prediction,
    )
)


final_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        final_prediction,
    )
)


final_r2 = (
    r2_score(
        y_test,
        final_prediction,
    )
)


print(
    "\n" + "=" * 70
)

print(
    "V18 FINAL TEST RESULTS"
)

print(
    "=" * 70
)

print(
    f"MAE : {final_mae:.4f}"
)

print(
    f"RMSE: {final_rmse:.4f}"
)

print(
    f"R²  : {final_r2:.4f}"
)


# ----------------------------------------------------------------
# 18. ACTIVE-HOURS PERFORMANCE
# ----------------------------------------------------------------

actual_active_mask = (
        y_test > ACTIVE_THRESHOLD
)


if actual_active_mask.sum() > 0:

    active_mae = (
        mean_absolute_error(
            y_test[
                actual_active_mask
            ],
            final_prediction[
                actual_active_mask
            ],
        )
    )

    active_rmse = np.sqrt(
        mean_squared_error(
            y_test[
                actual_active_mask
            ],
            final_prediction[
                actual_active_mask
            ],
        )
    )

    active_r2 = (
        r2_score(
            y_test[
                actual_active_mask
            ],
            final_prediction[
                actual_active_mask
            ],
        )
    )

else:

    active_mae = np.nan
    active_rmse = np.nan
    active_r2 = np.nan


print(
    "\nActive-hours only:"
)

print(
    f"MAE : {active_mae:.4f}"
)

print(
    f"RMSE: {active_rmse:.4f}"
)

print(
    f"R²  : {active_r2:.4f}"
)


# ----------------------------------------------------------------
# 19. PERSISTENCE BASELINE
# ----------------------------------------------------------------

print(
    "\n" + "=" * 70
)

print(
    "PERSISTENCE BASELINE"
)

print(
    "=" * 70
)


persistence_prediction = (
    test_data[
        "current_ac_power"
    ]
    .fillna(0.0)
    .to_numpy()
)


persistence_prediction = np.maximum(
    persistence_prediction,
    0.0,
)


persistence_mae = (
    mean_absolute_error(
        y_test,
        persistence_prediction,
    )
)


persistence_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        persistence_prediction,
    )
)


persistence_r2 = (
    r2_score(
        y_test,
        persistence_prediction,
    )
)


print(
    f"MAE : "
    f"{persistence_mae:.4f}"
)

print(
    f"RMSE: "
    f"{persistence_rmse:.4f}"
)

print(
    f"R²  : "
    f"{persistence_r2:.4f}"
)


# ----------------------------------------------------------------
# 20. COMPARISON WITH V15
# ----------------------------------------------------------------

V15_MAE = 20.1173
V15_RMSE = 41.5746
V15_R2 = 0.7778


print(
    "\n" + "=" * 70
)

print(
    "COMPARISON WITH V15"
)

print(
    "=" * 70
)

print(
    f"V15 MAE : "
    f"{V15_MAE:.4f}"
)

print(
    f"V18 MAE : "
    f"{final_mae:.4f}"
)

print(
    f"\nV15 RMSE: "
    f"{V15_RMSE:.4f}"
)

print(
    f"V18 RMSE: "
    f"{final_rmse:.4f}"
)

print(
    f"\nV15 R²  : "
    f"{V15_R2:.4f}"
)

print(
    f"V18 R²  : "
    f"{final_r2:.4f}"
)


# ----------------------------------------------------------------
# 21. VS PERSISTENCE
# ----------------------------------------------------------------

mae_change_vs_persistence = (
        (
                final_mae
                - persistence_mae
        )
        / persistence_mae
        * 100
)


rmse_change_vs_persistence = (
        (
                final_rmse
                - persistence_rmse
        )
        / persistence_rmse
        * 100
)


print(
    "\nChange vs persistence:"
)

print(
    f"MAE : "
    f"{mae_change_vs_persistence:+.2f}%"
)

print(
    f"RMSE: "
    f"{rmse_change_vs_persistence:+.2f}%"
)


# ----------------------------------------------------------------
# 22. SAVE PREDICTIONS
# ----------------------------------------------------------------

prediction_output = test_data[
    [
        "timestamp",
        "target_timestamp",
        target_col,
        "current_ac_power",
    ]
].copy()


prediction_output = (
    prediction_output.rename(
        columns={
            target_col:
                "actual_ac_power"
        }
    )
)


prediction_output[
    "stage1_active_probability"
] = test_probability


prediction_output[
    "stage1_active_prediction"
] = (
    test_active_prediction
)


prediction_output[
    "stage2_regression_prediction"
] = (
    regression_prediction
)


prediction_output[
    "v18_final_prediction"
] = (
    final_prediction
)


prediction_output[
    "persistence_prediction"
] = (
    persistence_prediction
)


prediction_output.to_csv(
    PREDICTION_FILE,
    index=False,
)


# ----------------------------------------------------------------
# 23. SAVE RESULTS
# ----------------------------------------------------------------

results_output = pd.DataFrame({

    "model": [
        "V18 Two-Stage Recency Weighted",
        "V15 Two-Stage",
        "Persistence",
    ],

    "mae": [
        final_mae,
        V15_MAE,
        persistence_mae,
    ],

    "rmse": [
        final_rmse,
        V15_RMSE,
        persistence_rmse,
    ],

    "r2": [
        final_r2,
        V15_R2,
        persistence_r2,
    ],

})


results_output.to_csv(
    RESULT_FILE,
    index=False,
)


# ----------------------------------------------------------------
# 24. FINAL
# ----------------------------------------------------------------

print(
    "\n" + "=" * 70
)

print(
    "FILES SAVED"
)

print(
    "=" * 70
)

print(
    f"\nPredictions:\n"
    f"{PREDICTION_FILE}"
)

print(
    f"\nResults:\n"
    f"{RESULT_FILE}"
)

print(
    "\n" + "=" * 70
)

print(
    "STEP 72 COMPLETE"
)

print(
    "=" * 70
)