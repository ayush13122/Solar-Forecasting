# Step 64 — V12 Ensemble Final Test.
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
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
# LOAD FORECASTING DATA
# ============================================================

df = pd.read_csv(forecast_path)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["target_timestamp"] = pd.to_datetime(
    df["target_timestamp"]
)

df = df.sort_values("timestamp").reset_index(drop=True)


print("=" * 70)
print("STEP 64 - MODEL 1 V12 FINAL TEST")
print("=" * 70)

print("\nInput shape:", df.shape)


# ============================================================
# MODEL 1 FEATURES
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
# FINAL TRAIN / TEST SPLIT
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


# ============================================================
# FINAL LIGHTGBM
# ============================================================

lgb_model = lgb.LGBMRegressor(
    objective="regression",
    metric="rmse",
    n_estimators=112,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbosity=-1
)


# ============================================================
# FINAL XGBOOST
# ============================================================

xgb_model = XGBRegressor(
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
# TRAIN BOTH
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FINAL V12 COMPONENTS")
print("=" * 70)

print("\nTraining LightGBM...")
lgb_model.fit(
    X_train,
    y_train
)

print("LightGBM completed.")


print("\nTraining XGBoost...")
xgb_model.fit(
    X_train,
    y_train,
    verbose=False
)

print("XGBoost completed.")


# ============================================================
# TEST PREDICTIONS
# ============================================================

lgb_test_pred = lgb_model.predict(X_test)

xgb_test_pred = xgb_model.predict(X_test)


# ============================================================
# V12 ENSEMBLE
# ============================================================

lgb_weight = 0.45
xgb_weight = 0.55


ensemble_test_pred = (
        lgb_weight * lgb_test_pred
        +
        xgb_weight * xgb_test_pred
)


# ============================================================
# METRICS FUNCTION
# ============================================================

def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    return mae, rmse, r2


# ============================================================
# COMPONENT METRICS
# ============================================================

lgb_mae, lgb_rmse, lgb_r2 = calculate_metrics(
    y_test,
    lgb_test_pred
)

xgb_mae, xgb_rmse, xgb_r2 = calculate_metrics(
    y_test,
    xgb_test_pred
)

ensemble_mae, ensemble_rmse, ensemble_r2 = calculate_metrics(
    y_test,
    ensemble_test_pred
)


# ============================================================
# PERSISTENCE BASELINE
# ============================================================

ac_df = pd.read_csv(
    preprocessed_path
)

ac_df["timestamp"] = pd.to_datetime(
    ac_df["timestamp"]
)

ac_df = ac_df[
    [
        "timestamp",
        "ac_power__5069"
    ]
].copy()


baseline_df = test_df[
    [
        "timestamp",
        "target_ac_power"
    ]
].merge(
    ac_df,
    on="timestamp",
    how="left"
)


baseline_pred = baseline_df[
    "ac_power__5069"
].values

baseline_actual = baseline_df[
    "target_ac_power"
].values


valid = (
        ~pd.isna(baseline_pred)
        &
        ~pd.isna(baseline_actual)
)


baseline_pred = baseline_pred[valid]
baseline_actual = baseline_actual[valid]


baseline_mae, baseline_rmse, baseline_r2 = calculate_metrics(
    baseline_actual,
    baseline_pred
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("MODEL 1 V12 FINAL TEST RESULTS")
print("=" * 70)

print("\nLightGBM component:")
print("MAE :", round(lgb_mae, 4))
print("RMSE:", round(lgb_rmse, 4))
print("R²  :", round(lgb_r2, 4))


print("\nXGBoost component:")
print("MAE :", round(xgb_mae, 4))
print("RMSE:", round(xgb_rmse, 4))
print("R²  :", round(xgb_r2, 4))


print("\nV12 Ensemble:")
print("MAE :", round(ensemble_mae, 4))
print("RMSE:", round(ensemble_rmse, 4))
print("R²  :", round(ensemble_r2, 4))


print("\nPersistence Baseline:")
print("MAE :", round(baseline_mae, 4))
print("RMSE:", round(baseline_rmse, 4))
print("R²  :", round(baseline_r2, 4))


# ============================================================
# REFERENCE MODEL 2
# ============================================================

print("\nModel 2 - Intraday:")
print("MAE :", 12.2133)
print("RMSE:", 24.3949)
print("R²  :", 0.9235)