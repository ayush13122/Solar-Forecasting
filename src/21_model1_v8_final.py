# Step 57 — V8 XGBoost ka final unseen test.
#
# Ab V8 ko rolling validation mein select kar liya hai, so ab 2017–2018 test first time use karenge.
from pathlib import Path

import numpy as np
import pandas as pd

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# FIND PROJECT ROOT
# ============================================================

current_path = Path(__file__).resolve()

PROJECT_ROOT = None

for parent in current_path.parents:
    if (parent / "data" / "processed").exists():
        PROJECT_ROOT = parent
        break

if PROJECT_ROOT is None:
    raise FileNotFoundError(
        "Could not locate project root containing data/processed."
    )


forecast_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "Model1_WeatherSolar_NextHour.csv"
)

preprocessed_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "PVDAQ_1433_ML_Ready_Preprocessed_Hourly.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(forecast_path)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["target_timestamp"] = pd.to_datetime(
    df["target_timestamp"]
)

df = df.sort_values("timestamp").reset_index(drop=True)


print("=" * 70)
print("STEP 57 - MODEL 1 V8 FINAL TEST EVALUATION")
print("=" * 70)

print("\nInput shape:", df.shape)


# ============================================================
# FEATURES
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
# FINAL RECENT-HISTORY TRAIN / TEST SPLIT
# ============================================================

train_start = pd.Timestamp("2014-01-01")
train_end = pd.Timestamp("2016-12-31 23:59:59")

test_start = pd.Timestamp("2017-01-01")


train_df = df[
    (df["target_timestamp"] >= train_start) &
    (df["target_timestamp"] <= train_end)
    ].copy()

test_df = df[
    df["target_timestamp"] >= test_start
    ].copy()


X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]


print("\nFinal split:")
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
# FINAL XGBOOST MODEL
# ============================================================

# Median best iteration from rolling validation:
# 161, 141, 222 -> median = 161

model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=161,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    tree_method="hist",
    n_jobs=-1
)


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FINAL MODEL 1 V8")
print("=" * 70)

model.fit(
    X_train,
    y_train,
    verbose=False
)

print("\nFinal V8 training completed.")
print("Trees:", model.n_estimators)


# ============================================================
# TEST PREDICTIONS
# ============================================================

test_pred = model.predict(X_test)


# ============================================================
# MODEL 1 V8 TEST METRICS
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
# PERSISTENCE BASELINE
# ============================================================

ac_df = pd.read_csv(preprocessed_path)

ac_df["timestamp"] = pd.to_datetime(
    ac_df["timestamp"]
)

ac_df = ac_df[
    [
        "timestamp",
        "ac_power__5069"
    ]
].copy()


test_baseline_df = test_df[
    [
        "timestamp",
        "target_ac_power"
    ]
].merge(
    ac_df,
    on="timestamp",
    how="left"
)


baseline_pred = (
    test_baseline_df["ac_power__5069"].values
)

baseline_actual = (
    test_baseline_df["target_ac_power"].values
)


valid_baseline = (
        ~pd.isna(baseline_pred)
        &
        ~pd.isna(baseline_actual)
)


baseline_pred = baseline_pred[
    valid_baseline
]

baseline_actual = baseline_actual[
    valid_baseline
]


baseline_mae = mean_absolute_error(
    baseline_actual,
    baseline_pred
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        baseline_actual,
        baseline_pred
    )
)

baseline_r2 = r2_score(
    baseline_actual,
    baseline_pred
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("MODEL 1 V8 FINAL TEST RESULTS")
print("=" * 70)

print("\nXGBoost V8:")
print("MAE :", round(test_mae, 4))
print("RMSE:", round(test_rmse, 4))
print("R²  :", round(test_r2, 4))


print("\nPersistence Baseline:")
print("MAE :", round(baseline_mae, 4))
print("RMSE:", round(baseline_rmse, 4))
print("R²  :", round(baseline_r2, 4))


# ============================================================
# COMPARISON WITH PREVIOUS LOCKED RESULTS
# ============================================================

print("\n" + "=" * 70)
print("REFERENCE RESULTS")
print("=" * 70)

print("\nModel 1 V1:")
print("MAE :", 18.6836)
print("RMSE:", 37.0633)
print("R²  :", 0.8234)

print("\nModel 1 V3:")
print("MAE :", 18.7105)
print("RMSE:", 37.8146)
print("R²  :", 0.8162)

print("\nModel 1 V4:")
print("MAE :", 18.8638)
print("RMSE:", 37.2375)
print("R²  :", 0.8217)

print("\nModel 1 V5:")
print("MAE :", 18.7334)
print("RMSE:", 38.3341)
print("R²  :", 0.8111)

print("\nModel 2:")
print("MAE :", 12.2133)
print("RMSE:", 24.3949)
print("R²  :", 0.9235)