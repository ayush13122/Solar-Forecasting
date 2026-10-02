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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

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
df["target_timestamp"] = pd.to_datetime(df["target_timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)


print("=" * 70)
print("STEP 51 - FINAL REFIT OF MODEL 1 V1")
print("=" * 70)

print("\nFull dataset:", df.shape)


# ============================================================
# MODEL 1 FEATURES - ORIGINAL 26
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


print("\nFeature count:", len(features))


# ============================================================
# FINAL TRAIN / TEST PERIOD
# ============================================================

train_end = pd.Timestamp("2016-12-31 23:59:59")
test_start = pd.Timestamp("2017-01-01")


# IMPORTANT:
# Split using TARGET timestamp so the target itself
# stays entirely inside the intended period.

train_df = df[
    df["target_timestamp"] <= train_end
    ].copy()

test_df = df[
    df["target_timestamp"] >= test_start
    ].copy()


# ============================================================
# CREATE X / y
# ============================================================

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]


print("\nFinal refit split:")
print("Train:", X_train.shape)
print("Test :", X_test.shape)

print(
    "\nTraining target period:",
    train_df["target_timestamp"].min(),
    "→",
    train_df["target_timestamp"].max()
)

print(
    "Test target period:",
    test_df["target_timestamp"].min(),
    "→",
    test_df["target_timestamp"].max()
)


# ============================================================
# FINAL MODEL PARAMETERS
# ============================================================

# Median best iteration from rolling validation:
# 171, 121, 258 → median = 171

params = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 171,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": 42,
    "verbosity": -1
}


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FINAL MODEL 1")
print("=" * 70)

model1_final = lgb.LGBMRegressor(**params)

model1_final.fit(
    X_train,
    y_train
)

print("\nFinal Model 1 training completed.")
print("Trees:", model1_final.n_estimators)


# ============================================================
# FINAL TEST PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL 1 TEST EVALUATION")
print("=" * 70)

test_pred = model1_final.predict(X_test)


# ============================================================
# METRICS
# ============================================================

test_mae = mean_absolute_error(
    y_test,
    test_pred
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_pred
    )
)

test_r2 = r2_score(
    y_test,
    test_pred
)


# ============================================================
# RESULTS
# ============================================================

print("\nMODEL 1 FINAL TEST RESULTS")

print("MAE :", round(test_mae, 4))
print("RMSE:", round(test_rmse, 4))
print("R²  :", round(test_r2, 4))


print("\nPERSISTENCE BASELINE")

print("MAE :", 17.6197)
print("RMSE:", 35.7158)
print("R²  :", 0.8360)


print("\nPREVIOUS MODEL 1 V1 TEST")

print("MAE :", 18.6836)
print("RMSE:", 37.0633)
print("R²  :", 0.8234)


print("\nMODEL 2 TEST")

print("MAE :", 12.2133)
print("RMSE:", 24.3949)
print("R²  :", 0.9235)