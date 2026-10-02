# Step 47 — Model 1 V4: Regularized LightGBM
# Step 47 — Model 1 V4
# Regularized LightGBM using original 26 features
# Expanding Window Validation

# Step 47 — Model 1 V4
# Regularized LightGBM using original 26 features

from pathlib import Path
import pandas as pd
import lightgbm as lgb


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
# LOAD DATA
# ============================================================

df = pd.read_csv(input_path)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)

# print("=" * 70)
# print("STEP 47 - MODEL 1 V4: REGULARIZED LIGHTGBM")
# print("=" * 70)
#
# print("\nInput shape:", df.shape)


# ============================================================
# MODEL 1 - ORIGINAL 26 FEATURES
# ============================================================

features = [
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

target = "target_ac_power"


# print("\nFeature count:", len(features))


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


X_train = train_df[features]
y_train = train_df[target]

X_val = val_df[features]
y_val = val_df[target]

X_test = test_df[features]
y_test = test_df[target]


# print("\nSplit sizes:")
# print("Train:", X_train.shape)
# print("Validation:", X_val.shape)
# print("Test:", X_test.shape)


# ============================================================
# V4 REGULARIZED PARAMETERS
# ============================================================

params = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 1500,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 8,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbosity": -1
}


# ============================================================
# TRAIN V4
# ============================================================

# print("\n" + "=" * 70)
# print("TRAINING MODEL 1 V4")
# print("=" * 70)

model1_v4 = lgb.LGBMRegressor(**params)

model1_v4.fit(
    X_train,
    y_train,
    eval_X=X_val,
    eval_y=y_val,
    callbacks=[
        lgb.early_stopping(75),
        lgb.log_evaluation(100)
    ]
)


# ============================================================
# VALIDATION PREDICTION
# ============================================================

v4_val_pred = model1_v4.predict(
    X_val,
    num_iteration=model1_v4.best_iteration_
)


# ============================================================
# VALIDATION METRICS
# ============================================================

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np


v4_val_mae = mean_absolute_error(
    y_val,
    v4_val_pred
)

v4_val_rmse = np.sqrt(
    mean_squared_error(
        y_val,
        v4_val_pred
    )
)

v4_val_r2 = r2_score(
    y_val,
    v4_val_pred
)


# ============================================================
# RESULT
# ============================================================

# print("\n" + "=" * 70)
# print("MODEL 1 V4 VALIDATION RESULT")
# print("=" * 70)
#
# print("Best iteration:",
#       model1_v4.best_iteration_)
#
# print("MAE:",
#       round(v4_val_mae, 4))
#
# print("RMSE:",
#       round(v4_val_rmse, 4))
#
# print("R²:",
#       round(v4_val_r2, 4))


# print("\nPrevious Model 1 validation results:")

# print("V1 RMSE:", 30.1396)



# Step 48 — V4 Final Test Evaluation
# ============================================================
# STEP 48 - FINAL TEST EVALUATION OF MODEL 1 V4
# ============================================================

print("\n" + "=" * 70)
print("STEP 48 - FINAL TEST EVALUATION OF MODEL 1 V4")
print("=" * 70)

# Test predictions
v4_test_pred = model1_v4.predict(
    X_test,
    num_iteration=model1_v4.best_iteration_
)

# Metrics
v4_test_mae = mean_absolute_error(
    y_test,
    v4_test_pred
)

v4_test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        v4_test_pred
    )
)

v4_test_r2 = r2_score(
    y_test,
    v4_test_pred
)


print("\nMODEL 1 V4 TEST RESULTS")

print("MAE :", round(v4_test_mae, 4))
print("RMSE:", round(v4_test_rmse, 4))
print("R²  :", round(v4_test_r2, 4))


print("\nREFERENCE TEST RESULTS")

print("\nPersistence:")
print("MAE :", 17.6197)
print("RMSE:", 35.7158)
print("R²  :", 0.8360)

print("\nModel 1 V1:")
print("MAE :", 18.6836)
print("RMSE:", 37.0633)
print("R²  :", 0.8234)

print("\nModel 1 V3:")
print("MAE :", 18.7105)
print("RMSE:", 37.8146)
print("R²  :", 0.8162)

print("\nModel 2:")
print("MAE :", 12.2133)
print("RMSE:", 24.3949)
print("R²  :", 0.9235)