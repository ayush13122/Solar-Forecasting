# Step 60 — V10 ka final unseen test.
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

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

history_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "PVDAQ_1433_ML_Ready_Preprocessed_Hourly.csv"
)


# ============================================================
# LOAD FORECAST DATA
# ============================================================

df = pd.read_csv(forecast_path)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["target_timestamp"] = pd.to_datetime(
    df["target_timestamp"]
)

df = df.sort_values("timestamp").reset_index(drop=True)


# ============================================================
# LOAD FULL HOURLY HISTORY
# ============================================================

history_df = pd.read_csv(history_path)

history_df["timestamp"] = pd.to_datetime(
    history_df["timestamp"]
)

history_df = history_df.sort_values(
    "timestamp"
).reset_index(drop=True)


print("=" * 70)
print("STEP 60 - MODEL 1 V10 FINAL TEST")
print("=" * 70)

print("\nForecast dataset:", df.shape)
print("History dataset:", history_df.shape)


# ============================================================
# CREATE TEMPORAL FEATURES
# ============================================================

hist = history_df[
    [
        "timestamp",
        "poa_irradiance__5061",
        "CLOUD_AMT",
        "ambient_temp__5062",
        "T2M"
    ]
].copy()


# ------------------------------------------------------------
# POA LAGS
# ------------------------------------------------------------

hist["poa_lag1"] = (
    hist["poa_irradiance__5061"].shift(1)
)

hist["poa_lag2"] = (
    hist["poa_irradiance__5061"].shift(2)
)

hist["poa_lag3"] = (
    hist["poa_irradiance__5061"].shift(3)
)


# ------------------------------------------------------------
# POA CHANGES
# ------------------------------------------------------------

hist["poa_change_1h"] = (
        hist["poa_irradiance__5061"]
        - hist["poa_lag1"]
)

hist["poa_change_3h"] = (
        hist["poa_irradiance__5061"]
        - hist["poa_lag3"]
)


# ------------------------------------------------------------
# ROLLING POA FEATURES
# ------------------------------------------------------------

hist["poa_rolling_mean_3h"] = (
    hist["poa_irradiance__5061"]
    .rolling(window=3, min_periods=1)
    .mean()
)

hist["poa_rolling_std_3h"] = (
    hist["poa_irradiance__5061"]
    .rolling(window=3, min_periods=2)
    .std()
)


# ------------------------------------------------------------
# CLOUD CHANGE
# ------------------------------------------------------------

hist["cloud_change_1h"] = (
        hist["CLOUD_AMT"]
        - hist["CLOUD_AMT"].shift(1)
)


# ------------------------------------------------------------
# TEMPERATURE LAGS
# ------------------------------------------------------------

hist["ambient_temp_lag1"] = (
    hist["ambient_temp__5062"].shift(1)
)

hist["T2M_lag1"] = (
    hist["T2M"].shift(1)
)


# ------------------------------------------------------------
# KEEP TEMPORAL FEATURES
# ------------------------------------------------------------

temporal_features = [
    "poa_lag1",
    "poa_lag2",
    "poa_lag3",
    "poa_change_1h",
    "poa_change_3h",
    "poa_rolling_mean_3h",
    "poa_rolling_std_3h",
    "cloud_change_1h",
    "ambient_temp_lag1",
    "T2M_lag1"
]

temporal_df = hist[
    ["timestamp"] + temporal_features
    ].copy()


# ============================================================
# MERGE TEMPORAL FEATURES
# ============================================================

df = df.merge(
    temporal_df,
    on="timestamp",
    how="left"
)


print("\nTemporal features:", len(temporal_features))
print("Total V10 features:", 26 + len(temporal_features))

print(
    "Missing temporal values:",
    df[temporal_features].isna().sum().sum()
)


# ============================================================
# BASE 26 FEATURES
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

features = base_features + temporal_features

target = "target_ac_power"


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


# ============================================================
# X / y
# ============================================================

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
# FINAL V10 MODEL
# ============================================================

# Rolling validation best iterations:
# 164, 126, 251
# Median = 164

model = lgb.LGBMRegressor(
    objective="regression",
    metric="rmse",
    n_estimators=164,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbosity=-1
)


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FINAL MODEL 1 V10")
print("=" * 70)

model.fit(
    X_train,
    y_train
)

print("\nFinal V10 training completed.")
print("Trees:", model.n_estimators)


# ============================================================
# TEST PREDICTIONS
# ============================================================

test_pred = model.predict(X_test)


# ============================================================
# MODEL 1 V10 TEST METRICS
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

ac_df = pd.read_csv(history_path)

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
print("MODEL 1 V10 FINAL TEST RESULTS")
print("=" * 70)

print("\nV10:")
print("MAE :", round(test_mae, 4))
print("RMSE:", round(test_rmse, 4))
print("R²  :", round(test_r2, 4))

print("\nPersistence:")
print("MAE :", round(baseline_mae, 4))
print("RMSE:", round(baseline_rmse, 4))
print("R²  :", round(baseline_r2, 4))


# ============================================================
# REFERENCE RESULTS
# ============================================================

print("\n" + "=" * 70)
print("REFERENCE RESULTS")
print("=" * 70)

print("\nModel 1 V1:")
print("MAE :", 18.6836)
print("RMSE:", 37.0633)
print("R²  :", 0.8234)

print("\nModel 1 V6:")
print("Rolling Avg MAE :", 14.7334)
print("Rolling Avg RMSE:", 30.0998)
print("Rolling Avg R²  :", 0.9097)

print("\nModel 2:")
print("MAE :", 12.2133)
print("RMSE:", 24.3949)
print("R²  :", 0.9235)