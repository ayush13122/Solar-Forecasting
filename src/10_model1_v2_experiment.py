from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import pvlib


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

print("=" * 70)
print("STEP 41 - MODEL 1 V2: TARGET-HOUR DETERMINISTIC FEATURES")
print("=" * 70)

print("\nInput shape:", df.shape)


# ============================================================
# TARGET-HOUR SOLAR POSITION
# ============================================================

target_mst = df["target_timestamp"].dt.tz_localize("Etc/GMT+7")

solar_position = pvlib.solarposition.get_solarposition(
    time=target_mst,
    latitude=39.7404,
    longitude=-105.1719
)

df["target_solar_zenith"] = solar_position["zenith"].values
df["target_solar_azimuth"] = solar_position["azimuth"].values


# ============================================================
# TARGET-HOUR TIME FEATURES
# ============================================================

target_hour = df["target_timestamp"].dt.hour
target_month = df["target_timestamp"].dt.month
target_day_of_year = df["target_timestamp"].dt.dayofyear

df["target_hour_sin"] = np.sin(
    2 * np.pi * target_hour / 24
)

df["target_hour_cos"] = np.cos(
    2 * np.pi * target_hour / 24
)

df["target_month_sin"] = np.sin(
    2 * np.pi * target_month / 12
)

df["target_month_cos"] = np.cos(
    2 * np.pi * target_month / 12
)

df["target_day_of_year_sin"] = np.sin(
    2 * np.pi * target_day_of_year / 365.25
)

df["target_day_of_year_cos"] = np.cos(
    2 * np.pi * target_day_of_year / 365.25
)


# ============================================================
# MODEL 1 V2 FEATURES
# ============================================================

current_weather_features = [
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
    "wind_direction_900_mb"
]

target_time_features = [
    "target_solar_zenith",
    "target_solar_azimuth",
    "target_hour_sin",
    "target_hour_cos",
    "target_month_sin",
    "target_month_cos",
    "target_day_of_year_sin",
    "target_day_of_year_cos"
]

model1_v2_features = (
        current_weather_features
        + target_time_features
)

target = "target_ac_power"

print("\nNumber of V2 features:", len(model1_v2_features))

print(
    "Missing values in V2 features:",
    df[model1_v2_features].isna().sum().sum()
)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

train_end = pd.Timestamp("2015-12-31 23:59:59")

val_start = pd.Timestamp("2016-01-01")
val_end = pd.Timestamp("2016-12-31 23:59:59")

test_start = pd.Timestamp("2017-01-01")


train_df = df[df["timestamp"] <= train_end].copy()

val_df = df[
    (df["timestamp"] >= val_start) &
    (df["timestamp"] <= val_end)
    ].copy()

test_df = df[df["timestamp"] >= test_start].copy()


X_train = train_df[model1_v2_features]
y_train = train_df[target]

X_val = val_df[model1_v2_features]
y_val = val_df[target]

X_test = test_df[model1_v2_features]
y_test = test_df[target]


print("\nV2 split:")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# ============================================================
# TRAIN MODEL 1 V2
# ============================================================

print("\n" + "=" * 70)
print("TRAINING MODEL 1 V2")
print("=" * 70)

params = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": 42,
    "verbosity": -1
}

model1_v2 = lgb.LGBMRegressor(**params)

model1_v2.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(50),
        lgb.log_evaluation(100)
    ]
)


# ============================================================
# RESULT
# ============================================================

print("\n" + "=" * 70)
print("MODEL 1 V2 RESULT")
print("=" * 70)

print("Best iteration:", model1_v2.best_iteration_)

print(
    "Validation RMSE:",
    round(model1_v2.best_score_["valid_0"]["rmse"], 4)
)

print("\nV1 Validation RMSE: 30.1396 kW")