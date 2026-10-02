# Step 42 → Model 1 V3: Irradiance Dynamics
# Step 43 → V3 Validation Evaluation
# Step 44 → V3 + POA Lag Missing Indicator

from pathlib import Path
import pandas as pd
import lightgbm as lgb
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

input_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "Model1_WeatherSolar_NextHour.csv"
)


# ============================================================
# LOAD MODEL 1 DATA
# ============================================================

df = pd.read_csv(input_path)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["target_timestamp"] = pd.to_datetime(df["target_timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)


# ============================================================
# LOAD FULL PREPROCESSED HOURLY DATA
# FOR HISTORICAL POA
# ============================================================

preprocessed_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "PVDAQ_1433_ML_Ready_Preprocessed_Hourly.csv"
)

history_df = pd.read_csv(preprocessed_path)

history_df["timestamp"] = pd.to_datetime(
    history_df["timestamp"]
)


# ============================================================
# CREATE PREVIOUS-HOUR POA
# USING TIMESTAMP MATCHING
# ============================================================

lag_df = history_df[
    [
        "timestamp",
        "poa_irradiance__5061"
    ]
].copy()

# Make previous-hour POA match current timestamp
lag_df["timestamp"] = (
        lag_df["timestamp"] + pd.Timedelta(hours=1)
)

lag_df = lag_df.rename(
    columns={
        "poa_irradiance__5061": "poa_irradiance_lag1"
    }
)

df = df.merge(
    lag_df,
    on="timestamp",
    how="left"
)


# ============================================================
# CREATE POA CHANGE
# ============================================================

df["poa_change_1h"] = (
        df["poa_irradiance__5061"]
        - df["poa_irradiance_lag1"]
)


# ============================================================
# CREATE POA LAG MISSING INDICATOR
# ============================================================

df["poa_lag1_missing"] = (
    df["poa_irradiance_lag1"]
    .isna()
    .astype(int)
)

print(
    "\nPOA lag missing indicator counts:"
)

print(
    df["poa_lag1_missing"].value_counts()
)
#
#
# print("\nPOA lag missing:",
#       df["poa_irradiance_lag1"].isna().sum())
#
# print("POA change missing:",
#       df["poa_change_1h"].isna().sum())


# ============================================================
# MODEL 1 BASE FEATURES
# ============================================================

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
    "day_of_year_cos"
]


# ============================================================
# V3 FEATURES
# ============================================================

v3_features = base_features + [
    "poa_irradiance_lag1",
    "poa_change_1h"
]

target = "target_ac_power"


# print("\nV3 feature count:", len(v3_features))
#
# print(
#     "Missing values in V3 features:",
#     df[v3_features].isna().sum().sum()
# )


# ============================================================
# IMPORTANT:
# DO NOT DROP ROWS WITH MISSING LAG
# LIGHTGBM CAN HANDLE MISSING VALUES
# ============================================================


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

train_end = pd.Timestamp("2015-12-31 23:59:59")

val_start = pd.Timestamp("2016-01-01")
val_end = pd.Timestamp("2016-12-31 23:59:59")

test_start = pd.Timestamp("2017-01-01")


train_df = df[
    df["timestamp"] <= train_end
    ].copy()

val_df = df[
    (df["timestamp"] >= val_start) &
    (df["timestamp"] <= val_end)
    ].copy()

test_df = df[
    df["timestamp"] >= test_start
    ].copy()


# ============================================================
# X / y
# ============================================================

X_train = train_df[v3_features]
y_train = train_df[target]

X_val = val_df[v3_features]
y_val = val_df[target]

X_test = test_df[v3_features]
y_test = test_df[target]


# ============================================================
# SPLIT VERIFICATION
# ============================================================

# print("\nV3 split:")
# print("Train:", X_train.shape)
# print("Validation:", X_val.shape)
# print("Test:", X_test.shape)
#
# print("\nMissing values:")
# print("Train:", X_train.isna().sum().sum())
# print("Validation:", X_val.isna().sum().sum())
# print("Test:", X_test.isna().sum().sum())


# ============================================================
# TRAIN MODEL 1 V3 + MISSING INDICATOR
# ============================================================

# print("\n" + "=" * 70)
# print("TRAINING MODEL 1 V3 + MISSING INDICATOR")
# print("=" * 70)


params = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": 42,
    "verbosity": -1
}


model1_v3 = lgb.LGBMRegressor(**params)


model1_v3.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(50),
        lgb.log_evaluation(100)
    ]
)


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

v3_val_pred = model1_v3.predict(
    X_val,
    num_iteration=model1_v3.best_iteration_
)


# ============================================================
# VALIDATION METRICS
# ============================================================

v3_val_mae = mean_absolute_error(
    y_val,
    v3_val_pred
)

v3_val_rmse = np.sqrt(
    mean_squared_error(
        y_val,
        v3_val_pred
    )
)

v3_val_r2 = r2_score(
    y_val,
    v3_val_pred
)


# ============================================================
# FINAL VALIDATION RESULT
# ============================================================

# print("\n" + "=" * 70)
# print("MODEL 1 V3 + MISSING INDICATOR RESULT")
# print("=" * 70)
#
# print("Best iteration:",
#       model1_v3.best_iteration_)
#
# print("MAE:",
#       round(v3_val_mae, 4))
#
# print("RMSE:",
#       round(v3_val_rmse, 4))
#
# print("R²:",
#       round(v3_val_r2, 4))
#
#
# print("\nPrevious Model 1 results:")
# print("V1 RMSE:", 30.1396)
# print("V2 RMSE:", 30.0928)
# print("V3 RMSE:", 29.3956)


# Ab important Step 45
print("\n" + "=" * 70)
print("STEP 45 - V3 VS PERSISTENCE ON VALIDATION")
print("=" * 70)

# Load original preprocessed data for current-hour AC
history_ac_df = pd.read_csv(preprocessed_path)

history_ac_df["timestamp"] = pd.to_datetime(
    history_ac_df["timestamp"]
)

history_ac_df = history_ac_df[
    ["timestamp", "ac_power__5069"]
].copy()


# Attach current-hour AC to V3 validation timestamps
val_compare = val_df[
    ["timestamp", "target_ac_power"]
].merge(
    history_ac_df,
    on="timestamp",
    how="left"
)


# Persistence prediction = current-hour AC
validation_baseline_pred = (
    val_compare["ac_power__5069"]
    .values
)

validation_actual = (
    val_compare["target_ac_power"]
    .values
)


# Keep only valid baseline rows
valid_baseline = ~pd.isna(validation_baseline_pred)

validation_baseline_pred = validation_baseline_pred[
    valid_baseline
]

validation_actual = validation_actual[
    valid_baseline
]


# V3 predictions on same valid rows
v3_compare_pred = v3_val_pred[valid_baseline]


# Metrics
baseline_val_mae = mean_absolute_error(
    validation_actual,
    validation_baseline_pred
)

baseline_val_rmse = np.sqrt(
    mean_squared_error(
        validation_actual,
        validation_baseline_pred
    )
)

baseline_val_r2 = r2_score(
    validation_actual,
    validation_baseline_pred
)


v3_compare_mae = mean_absolute_error(
    validation_actual,
    v3_compare_pred
)

v3_compare_rmse = np.sqrt(
    mean_squared_error(
        validation_actual,
        v3_compare_pred
    )
)

v3_compare_r2 = r2_score(
    validation_actual,
    v3_compare_pred
)


# print("\nValidation comparison:")
#
# print("\nPersistence:")
# print("MAE :", round(baseline_val_mae, 4))
# print("RMSE:", round(baseline_val_rmse, 4))
# print("R²  :", round(baseline_val_r2, 4))
#
# print("\nModel 1 V3:")
# print("MAE :", round(v3_compare_mae, 4))
# print("RMSE:", round(v3_compare_rmse, 4))
# print("R²  :", round(v3_compare_r2, 4))
#
# print("\nValidation rows compared:", len(validation_actual))


# ============================================================
# STEP 46 - FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 46 - FINAL TEST EVALUATION OF MODEL 1 V3")
print("=" * 70)

# Test predictions
v3_test_pred = model1_v3.predict(
    X_test,
    num_iteration=model1_v3.best_iteration_
)

# Test metrics
v3_test_mae = mean_absolute_error(
    y_test,
    v3_test_pred
)

v3_test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        v3_test_pred
    )
)

v3_test_r2 = r2_score(
    y_test,
    v3_test_pred
)


print("\nMODEL 1 V3 TEST RESULTS")

print("MAE :", round(v3_test_mae, 4))
print("RMSE:", round(v3_test_rmse, 4))
print("R²  :", round(v3_test_r2, 4))


print("\nREFERENCE TEST RESULTS")

print("\nPersistence:")
print("MAE :", 17.6197)
print("RMSE:", 35.7158)
print("R²  :", 0.8360)

print("\nModel 1 V1:")
print("MAE :", 18.6836)
print("RMSE:", 37.0633)
print("R²  :", 0.8234)

print("\nModel 2:")
print("MAE :", 12.2133)
print("RMSE:", 24.3949)
print("R²  :", 0.9235)