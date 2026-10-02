# Step 69 — V15 Final Test
# ================================================================
# STEP 69 - MODEL 1 V15 FINAL TEST
# ================================================================
# Final confirmatory evaluation of selected V15 Two-Stage model
#
# Train: 2014-2016
# Test : 2017-2018
#
# Stage 1:
#   Active generation / Near-zero classification
#
# Stage 2:
#   Active-hour AC power regression
#
# Final:
#   If classifier -> Near-zero : 0 kW
#   If classifier -> Active     : Stage-2 prediction
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
print("STEP 69 - MODEL 1 V15 FINAL TEST")
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

    # TensorFlow environment also counts
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

PREDICTION_FILE = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "Model1_V15_Final_Test_Predictions.csv"
)

RESULT_FILE = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "Model1_V15_Final_Test_Results.csv"
)


# ----------------------------------------------------------------
# 3. SETTINGS
# ----------------------------------------------------------------

ACTIVE_THRESHOLD = 1.0

CLASSIFICATION_THRESHOLD = 0.50

# Average best iterations from V15 rolling validation:
#
# Classifier:
#   125, 84, 103
#   average = 104
#
# Regressor:
#   117, 195, 367
#   average = 226

CLASSIFIER_ESTIMATORS = 104

REGRESSOR_ESTIMATORS = 226


# ----------------------------------------------------------------
# 4. LOAD DATA
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
    parse_dates=[
        "timestamp"
    ]
)

history_df = history_df.sort_values(
    "timestamp"
).reset_index(drop=True)


print(
    f"\nForecast dataset: {forecast_df.shape}"
)

print(
    f"History dataset : {history_df.shape}"
)


# ----------------------------------------------------------------
# 5. BASE FEATURES
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
# 6. CREATE V14/V15 PHYSICS FEATURES
# ----------------------------------------------------------------

print(
    "\nCreating physics-informed features..."
)


# Solar elevation
history_df["solar_elevation_deg"] = (
        90.0
        - history_df["solar_zenith_angle"]
)


# Cosine of solar zenith
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


v15_features = (
        base_features
        + physics_features
)


print(
    f"\nBase features    : "
    f"{len(base_features)}"
)

print(
    f"Physics features : "
    f"{len(physics_features)}"
)

print(
    f"Total V15 features: "
    f"{len(v15_features)}"
)


# ----------------------------------------------------------------
# 7. MERGE PHYSICS FEATURES
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
# 8. ADD CURRENT AC FOR PERSISTENCE BASELINE ONLY
# ----------------------------------------------------------------

current_ac_source = history_df[
    [
        "timestamp",
        "ac_power__5069"
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
    validate="one_to_one"
)


# ----------------------------------------------------------------
# 9. MISSING CHECK
# ----------------------------------------------------------------

missing_features = (
    forecast_df[v15_features]
    .isna()
    .sum()
)

print(
    "\nMissing V15 feature values:"
)

print(
    missing_features[
        missing_features > 0
        ]
)

if missing_features.sum() > 0:
    raise ValueError(
        "Missing V15 features detected."
    )


# ----------------------------------------------------------------
# 10. CREATE STAGE-1 TARGET
# ----------------------------------------------------------------

forecast_df["active_target"] = (
        forecast_df[target_col]
        > ACTIVE_THRESHOLD
).astype(int)


# ----------------------------------------------------------------
# 11. DEFINE TRAIN / TEST
# ----------------------------------------------------------------

train_mask = (
        (forecast_df["target_timestamp"]
         >= pd.Timestamp("2014-01-01"))
        &
        (forecast_df["target_timestamp"]
         <= pd.Timestamp("2016-12-31 23:59:59"))
        &
        forecast_df[target_col].notna()
)

test_mask = (
        (forecast_df["target_timestamp"]
         >= pd.Timestamp("2017-01-01"))
        &
        (forecast_df["target_timestamp"]
         <= pd.Timestamp("2018-12-31 23:59:59"))
        &
        forecast_df[target_col].notna()
)


train_data = forecast_df.loc[
    train_mask
].copy()

test_data = forecast_df.loc[
    test_mask
].copy()


print("\n" + "-" * 70)
print("FINAL TRAIN / TEST SPLIT")
print("-" * 70)

print(
    f"Train rows: {len(train_data)}"
)

print(
    f"Test rows : {len(test_data)}"
)

print(
    f"Train target period: "
    f"{train_data['target_timestamp'].min()} "
    f"→ "
    f"{train_data['target_timestamp'].max()}"
)

print(
    f"Test target period : "
    f"{test_data['target_timestamp'].min()} "
    f"→ "
    f"{test_data['target_timestamp'].max()}"
)


# ----------------------------------------------------------------
# 12. STAGE 1 - CLASSIFIER
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("STAGE 1 - ACTIVE / NEAR-ZERO CLASSIFIER")
print("=" * 70)


X_train_cls = train_data[
    v15_features
]

y_train_cls = train_data[
    "active_target"
]


X_test_cls = test_data[
    v15_features
]

y_test_cls = test_data[
    "active_target"
]


classifier = lgb.LGBMClassifier(

    objective="binary",

    n_estimators=CLASSIFIER_ESTIMATORS,

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
    y_train_cls
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
        test_active_prediction
    )
)

classification_precision = (
    precision_score(
        y_test_cls,
        test_active_prediction,
        zero_division=0
    )
)

classification_recall = (
    recall_score(
        y_test_cls,
        test_active_prediction,
        zero_division=0
    )
)

classification_f1 = (
    f1_score(
        y_test_cls,
        test_active_prediction,
        zero_division=0
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
# 13. STAGE 2 - ACTIVE POWER REGRESSOR
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("STAGE 2 - ACTIVE POWER REGRESSOR")
print("=" * 70)


active_train = train_data[
    train_data[
        "active_target"
    ] == 1
    ].copy()


X_train_reg = active_train[
    v15_features
]

y_train_reg = active_train[
    target_col
]


print(
    f"\nActive training rows: "
    f"{len(active_train)}"
)


regressor = lgb.LGBMRegressor(

    objective="regression",

    n_estimators=REGRESSOR_ESTIMATORS,

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
    y_train_reg
)


# Predict Stage-2 power for every test row
regression_prediction = (
    regressor.predict(
        X_test_cls
    )
)


# Physical lower bound
regression_prediction = np.maximum(
    regression_prediction,
    0.0
)


# ----------------------------------------------------------------
# 14. FINAL TWO-STAGE PREDICTION
# ----------------------------------------------------------------

final_prediction = np.where(

    test_active_prediction == 1,

    regression_prediction,

    0.0
)


final_prediction = np.maximum(
    final_prediction,
    0.0
)


# ----------------------------------------------------------------
# 15. MODEL METRICS
# ----------------------------------------------------------------

y_test = test_data[
    target_col
].to_numpy()


final_mae = (
    mean_absolute_error(
        y_test,
        final_prediction
    )
)

final_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        final_prediction
    )
)

final_r2 = (
    r2_score(
        y_test,
        final_prediction
    )
)


print("\n" + "=" * 70)
print("V15 FINAL TEST RESULTS")
print("=" * 70)

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
# 16. ACTIVE-ONLY TEST PERFORMANCE
# ----------------------------------------------------------------

actual_active_mask = (
        y_test > ACTIVE_THRESHOLD
)


if actual_active_mask.sum() > 0:

    active_only_mae = (
        mean_absolute_error(
            y_test[actual_active_mask],
            final_prediction[
                actual_active_mask
            ]
        )
    )

    active_only_rmse = np.sqrt(
        mean_squared_error(
            y_test[actual_active_mask],
            final_prediction[
                actual_active_mask
            ]
        )
    )

    active_only_r2 = (
        r2_score(
            y_test[actual_active_mask],
            final_prediction[
                actual_active_mask
            ]
        )
    )

else:

    active_only_mae = np.nan
    active_only_rmse = np.nan
    active_only_r2 = np.nan


print("\nActive-hours only:")

print(
    f"MAE : {active_only_mae:.4f}"
)

print(
    f"RMSE: {active_only_rmse:.4f}"
)

print(
    f"R²  : {active_only_r2:.4f}"
)


# ----------------------------------------------------------------
# 17. PERSISTENCE BASELINE
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("PERSISTENCE BASELINE")
print("=" * 70)


persistence_prediction = (
    test_data[
        "current_ac_power"
    ]
    .fillna(0.0)
    .to_numpy()
)


persistence_prediction = np.maximum(
    persistence_prediction,
    0.0
)


persistence_mae = (
    mean_absolute_error(
        y_test,
        persistence_prediction
    )
)

persistence_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        persistence_prediction
    )
)

persistence_r2 = (
    r2_score(
        y_test,
        persistence_prediction
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
# 18. IMPROVEMENT VS PERSISTENCE
# ----------------------------------------------------------------

mae_improvement = (
        (
                persistence_mae
                - final_mae
        )
        / persistence_mae
        * 100
)

rmse_improvement = (
        (
                persistence_rmse
                - final_rmse
        )
        / persistence_rmse
        * 100
)


print("\nImprovement vs persistence:")

print(
    f"MAE improvement : "
    f"{mae_improvement:.2f}%"
)

print(
    f"RMSE improvement: "
    f"{rmse_improvement:.2f}%"
)


# ----------------------------------------------------------------
# 19. SAVE PREDICTIONS
# ----------------------------------------------------------------

prediction_output = test_data[
    [
        "timestamp",
        "target_timestamp",
        target_col,
        "current_ac_power",
    ]
].copy()


prediction_output = prediction_output.rename(

    columns={
        target_col:
            "actual_ac_power"
    }
)


prediction_output[
    "stage1_active_probability"
] = test_probability


prediction_output[
    "stage1_active_prediction"
] = test_active_prediction


prediction_output[
    "stage2_regression_prediction"
] = regression_prediction


prediction_output[
    "v15_final_prediction"
] = final_prediction


prediction_output[
    "persistence_prediction"
] = persistence_prediction


prediction_output.to_csv(
    PREDICTION_FILE,
    index=False
)


# ----------------------------------------------------------------
# 20. SAVE METRICS
# ----------------------------------------------------------------

results = pd.DataFrame({

    "model": [
        "V15 Two-Stage",
        "Persistence"
    ],

    "mae": [
        final_mae,
        persistence_mae
    ],

    "rmse": [
        final_rmse,
        persistence_rmse
    ],

    "r2": [
        final_r2,
        persistence_r2
    ],

})


results.to_csv(
    RESULT_FILE,
    index=False
)


print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    f"\nPredictions:\n{PREDICTION_FILE}"
)

print(
    f"\nResults:\n{RESULT_FILE}"
)


print("\n" + "=" * 70)
print("STEP 69 COMPLETE")
print("=" * 70)