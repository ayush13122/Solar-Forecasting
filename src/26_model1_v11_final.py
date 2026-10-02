# Step 62 — V11 Final Test
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
print("STEP 62 - MODEL 1 V11 FINAL TEST")
print("=" * 70)

print("\nInput shape:", df.shape)


# ============================================================
# BASE FEATURES
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

target = "target_ac_power"


# ============================================================
# CLEAR-SKY FEATURES
# ============================================================

mst_time = pd.DatetimeIndex(
    df["timestamp"]
).tz_localize("Etc/GMT+7")


location = pvlib.location.Location(
    latitude=39.7404,
    longitude=-105.1719,
    tz="Etc/GMT+7"
)


solar_position = location.get_solarposition(
    mst_time
)


clear_sky = location.get_clearsky(
    mst_time,
    model="ineichen",
    solar_position=solar_position
)


df["clear_sky_ghi"] = clear_sky["ghi"].values


# ============================================================
# CLEARNess INDEX
# ============================================================

df["clearness_index"] = 0.0

valid_clear_sky = (
        df["clear_sky_ghi"] > 20
)

df.loc[
    valid_clear_sky,
    "clearness_index"
] = (
        df.loc[
            valid_clear_sky,
            "ALLSKY_SFC_SW_DWN"
        ]
        /
        df.loc[
            valid_clear_sky,
            "clear_sky_ghi"
        ]
)


print("\nClearness index:")
print(df["clearness_index"].describe())

print(
    "\nClearness index > 2:",
    (df["clearness_index"] > 2).sum()
)

print(
    "Clearness index > 5:",
    (df["clearness_index"] > 5).sum()
)


# ============================================================
# V11 FEATURES
# ============================================================

features = base_features + [
    "clear_sky_ghi",
    "clearness_index"
]


print(
    "\nTotal V11 features:",
    len(features)
)


# ============================================================
# FINAL TRAIN / TEST SPLIT
# ============================================================

train_start = pd.Timestamp("2014-01-01")
train_end = pd.Timestamp(
    "2016-12-31 23:59:59"
)

test_start = pd.Timestamp("2017-01-01")


train_df = df[
    (df["target_timestamp"] >= train_start)
    &
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
# FINAL V11 MODEL
# ============================================================

model = lgb.LGBMRegressor(
    objective="regression",
    metric="rmse",
    n_estimators=150,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbosity=-1
)


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FINAL MODEL 1 V11")
print("=" * 70)

model.fit(
    X_train,
    y_train
)

print("\nTraining completed.")
print("Trees:", model.n_estimators)


# ============================================================
# TEST PREDICTION
# ============================================================

test_pred = model.predict(
    X_test
)


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
print("MODEL 1 V11 FINAL TEST RESULTS")
print("=" * 70)

print("\nV11:")
print("MAE :", round(test_mae, 4))
print("RMSE:", round(test_rmse, 4))
print("R²  :", round(test_r2, 4))


print("\nPersistence:")
print("MAE :", round(baseline_mae, 4))
print("RMSE:", round(baseline_rmse, 4))
print("R²  :", round(baseline_r2, 4))


print("\nReference - Model 2:")
print("MAE :", 12.2133)
print("RMSE:", 24.3949)
print("R²  :", 0.9235)