# Step 53 — V5 ka final unseen test evaluation.
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import pvlib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PROJECT ROOT
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


input_path = (
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

df = pd.read_csv(input_path)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["target_timestamp"] = pd.to_datetime(df["target_timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)


print("=" * 70)
print("STEP 53 - MODEL 1 V5 FINAL TEST EVALUATION")
print("=" * 70)

print("\nInput shape:", df.shape)


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

dc_capacity_kw = 449.28


print("\nFeature count:", len(features))


# ============================================================
# TARGET-HOUR SOLAR POSITION
# ============================================================

target_mst = df["target_timestamp"].dt.tz_localize(
    "Etc/GMT+7"
)

target_solar_position = pvlib.solarposition.get_solarposition(
    time=target_mst,
    latitude=39.7404,
    longitude=-105.1719
)

df["target_solar_zenith"] = (
    target_solar_position["zenith"].values
)


# ============================================================
# DAYLIGHT / NIGHT
# ============================================================

df["target_is_daylight"] = (
        df["target_solar_zenith"] < 90
).astype(int)


# ============================================================
# NORMALIZED TARGET
# ============================================================

df["target_power_normalized"] = (
        df[target] / dc_capacity_kw
)


# ============================================================
# FINAL TRAIN / TEST PERIOD
# ============================================================

train_end = pd.Timestamp("2016-12-31 23:59:59")
test_start = pd.Timestamp("2017-01-01")


train_df = df[
    df["target_timestamp"] <= train_end
    ].copy()

test_df = df[
    df["target_timestamp"] >= test_start
    ].copy()


print("\nFinal split:")
print("Train:", train_df.shape)
print("Test :", test_df.shape)


# ============================================================
# DAYLIGHT TRAINING DATA
# ============================================================

train_daylight = train_df[
    train_df["target_is_daylight"] == 1
    ].copy()


print("\nTraining rows:")
print("All:", len(train_df))
print("Daylight:", len(train_daylight))


# ============================================================
# TRAINING DATA
# ============================================================

X_train = train_daylight[features]

y_train = train_daylight[
    "target_power_normalized"
]


# ============================================================
# FINAL V5 PARAMETERS
# ============================================================

# Median best iteration from rolling validation:
# 116, 105, 251 -> median = 116

params = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 116,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": 42,
    "verbosity": -1
}


# ============================================================
# TRAIN FINAL V5
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FINAL MODEL 1 V5")
print("=" * 70)

model1_v5_final = lgb.LGBMRegressor(
    **params
)

model1_v5_final.fit(
    X_train,
    y_train
)

print("\nFinal V5 training completed.")
print("Trees:", model1_v5_final.n_estimators)


# ============================================================
# CREATE ALL-HOURS TEST PREDICTIONS
# ============================================================

test_daylight_mask = (
        test_df["target_is_daylight"].values == 1
)

test_pred_normalized = np.zeros(
    len(test_df)
)


# ------------------------------------------------------------
# Predict only during daylight
# ------------------------------------------------------------

if test_daylight_mask.sum() > 0:

    daylight_test = test_df.loc[
        test_daylight_mask,
        features
    ]

    test_pred_normalized[
        test_daylight_mask
    ] = model1_v5_final.predict(
        daylight_test
    )


# ------------------------------------------------------------
# Convert normalized prediction back to kW
# ------------------------------------------------------------

test_pred = (
        test_pred_normalized
        * dc_capacity_kw
)


actual = test_df[target].values


# ============================================================
# METRICS
# ============================================================

test_mae = mean_absolute_error(
    actual,
    test_pred
)

test_rmse = np.sqrt(
    mean_squared_error(
        actual,
        test_pred
    )
)

test_r2 = r2_score(
    actual,
    test_pred
)


# ============================================================
# DAYLIGHT-ONLY METRICS
# ============================================================

daylight_actual = actual[
    test_daylight_mask
]

daylight_pred = test_pred[
    test_daylight_mask
]


daylight_mae = mean_absolute_error(
    daylight_actual,
    daylight_pred
)

daylight_rmse = np.sqrt(
    mean_squared_error(
        daylight_actual,
        daylight_pred
    )
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
    test_baseline_df["ac_power__5069"]
    .values
)

baseline_actual = (
    test_baseline_df["target_ac_power"]
    .values
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
print("MODEL 1 V5 FINAL TEST RESULTS")
print("=" * 70)

print("\nAll-hours:")
print("MAE :", round(test_mae, 4))
print("RMSE:", round(test_rmse, 4))
print("R²  :", round(test_r2, 4))

print("\nDaylight-only:")
print("MAE :", round(daylight_mae, 4))
print("RMSE:", round(daylight_rmse, 4))

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

print("\nModel 2:")
print("MAE :", 12.2133)
print("RMSE:", 24.3949)
print("R²  :", 0.9235)