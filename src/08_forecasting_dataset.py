
# Step 28 — New script
from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================
import pandas as pd
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent

input_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "PVDAQ_1433_ML_Ready_Preprocessed_Hourly.csv"
)

df = pd.read_csv(input_path)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)


# print("=" * 70)
# print("STEP 28 - CREATE DUAL-HORIZON FORECASTING DATASETS")
# print("=" * 70)
#
# print("\nInput dataset shape:", df.shape)
# print("Start:", df["timestamp"].min())
# print("End  :", df["timestamp"].max())


# ============================================================
# COMMON 26 FEATURES
# ============================================================

weather_solar_features = [
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

target_col = "ac_power__5069"


# ============================================================
# CREATE NEXT-HOUR TARGET
# ============================================================

df["target_timestamp"] = df["timestamp"].shift(-1)
df["target_ac_power"] = df[target_col].shift(-1)

time_gap = df["target_timestamp"] - df["timestamp"]

valid_1h = time_gap == pd.Timedelta(hours=1)

# print("\nTotal candidate rows:", len(df))
# print("Valid 1-hour pairs:", valid_1h.sum())
# print("Invalid time pairs:", (~valid_1h).sum())


# Keep only genuine t -> t+1 pairs
forecast_base = df.loc[valid_1h].copy()


# ============================================================
# MODEL 1 - WEATHER/SOLAR MODEL
# ============================================================

model1_cols = (
        ["timestamp"]
        + weather_solar_features
        + ["target_timestamp", "target_ac_power"]
)

model1_df = forecast_base[model1_cols].copy()


# ============================================================
# MODEL 2 - INTRADAY MODEL
# ============================================================

model2_features = ["ac_power__5069"] + weather_solar_features

model2_cols = (
        ["timestamp"]
        + model2_features
        + ["target_timestamp", "target_ac_power"]
)

model2_df = forecast_base[model2_cols].copy()


# ============================================================
# BASIC VERIFICATION
# ============================================================

# print("\n" + "-" * 70)
# print("MODEL 1 - WEATHER/SOLAR FORECASTING")
# print("-" * 70)
#
# print("Rows:", len(model1_df))
# print("Features:", len(weather_solar_features))
# print("Missing values:", model1_df.isna().sum().sum())
#
#
# print("\n" + "-" * 70)
# print("MODEL 2 - INTRA-DAY FORECASTING")
# print("-" * 70)
#
# print("Rows:", len(model2_df))
# print("Features:", len(model2_features))
# print("Missing values:", model2_df.isna().sum().sum())


# ============================================================
# SAVE DATASETS
# ============================================================

model1_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "Model1_WeatherSolar_NextHour.csv"
)

model2_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "Model2_Intraday_NextHour.csv"
)

model1_df.to_csv(model1_path, index=False)
model2_df.to_csv(model2_path, index=False)

#
# print("\n" + "=" * 70)
# print("DATASETS SAVED")
# print("=" * 70)
#
# print("\nModel 1:", model1_path)
# print("Model 2:", model2_path)
#
# print("\nModel 1 shape:", model1_df.shape)
# print("Model 2 shape:", model2_df.shape)


# Step 29 — Chronological Split
# print("\n" + "=" * 70)
# print("STEP 29 - CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT")
# print("=" * 70)
#
# # Model 1 and Model 2 have the same timestamps
# print("\nModel 1 date range:")
# print("Start:", model1_df["timestamp"].min())
# print("End  :", model1_df["timestamp"].max())
#
# print("\nModel 2 date range:")
# print("Start:", model2_df["timestamp"].min())
# print("End  :", model2_df["timestamp"].max())


# print("\nYear-wise record count:")

year_counts = (
    model1_df["timestamp"]
    .dt.year
    .value_counts()
    .sort_index()
)

# print(year_counts)
#
#
# print("\nYear-wise date coverage:")

year_summary = (
    model1_df
    .groupby(model1_df["timestamp"].dt.year)["timestamp"]
    .agg(["min", "max", "count"])
)

# print(year_summary)

# Step 30 — Same Chronological Split for Both Models

# print("\n" + "=" * 70)
# print("STEP 30 - CREATE CHRONOLOGICAL SPLITS")
# print("=" * 70)

# Same time boundaries for both models
train_start = pd.Timestamp("2010-01-01")
train_end = pd.Timestamp("2015-12-31 23:59:59")

val_start = pd.Timestamp("2016-01-01")
val_end = pd.Timestamp("2016-12-31 23:59:59")

test_start = pd.Timestamp("2017-01-01")
test_end = pd.Timestamp("2018-12-31 23:59:59")


# -----------------------------
# MODEL 1 SPLIT
# -----------------------------

model1_train = model1_df[
    (model1_df["timestamp"] >= train_start) &
    (model1_df["timestamp"] <= train_end)
    ].copy()

model1_val = model1_df[
    (model1_df["timestamp"] >= val_start) &
    (model1_df["timestamp"] <= val_end)
    ].copy()

model1_test = model1_df[
    (model1_df["timestamp"] >= test_start) &
    (model1_df["timestamp"] <= test_end)
    ].copy()


# -----------------------------
# MODEL 2 SPLIT
# -----------------------------

model2_train = model2_df[
    (model2_df["timestamp"] >= train_start) &
    (model2_df["timestamp"] <= train_end)
    ].copy()

model2_val = model2_df[
    (model2_df["timestamp"] >= val_start) &
    (model2_df["timestamp"] <= val_end)
    ].copy()

model2_test = model2_df[
    (model2_df["timestamp"] >= test_start) &
    (model2_df["timestamp"] <= test_end)
    ].copy()


# -----------------------------
# VERIFY
# -----------------------------

# print("\nMODEL 1")
# print("Train:", model1_train.shape)
# print("Validation:", model1_val.shape)
# print("Test:", model1_test.shape)
#
# print("\nMODEL 2")
# print("Train:", model2_train.shape)
# print("Validation:", model2_val.shape)
# print("Test:", model2_test.shape)
#
#
# print("\nDate ranges:")
#
# print(
#     "Model 1 Train:",
#     model1_train["timestamp"].min(),
#     "→",
#     model1_train["timestamp"].max()
# )

# print(
#     "Model 1 Validation:",
#     model1_val["timestamp"].min(),
#     "→",
#     model1_val["timestamp"].max()
# )
#
# print(
#     "Model 1 Test:",
#     model1_test["timestamp"].min(),
#     "→",
#     model1_test["timestamp"].max()
# )
#
# print(
#     "Model 2 Train:",
#     model2_train["timestamp"].min(),
#     "→",
#     model2_train["timestamp"].max()
# )

# print(
#     "Model 2 Validation:",
#     model2_val["timestamp"].min(),
#     "→",
#     model2_val["timestamp"].max()
# )
#
# print(
#     "Model 2 Test:",
#     model2_test["timestamp"].min(),
#     "→",
#     model2_test["timestamp"].max()
# )


# Step 31 — Final X/y for Both Models
# print("\n" + "=" * 70)
# print("STEP 31 - FINALIZE X/y FOR BOTH MODELS")
# print("=" * 70)


# ============================================================
# MODEL 1
# ============================================================

model1_X_train = model1_train[weather_solar_features].copy()
model1_y_train = model1_train["target_ac_power"].copy()

model1_X_val = model1_val[weather_solar_features].copy()
model1_y_val = model1_val["target_ac_power"].copy()

model1_X_test = model1_test[weather_solar_features].copy()
model1_y_test = model1_test["target_ac_power"].copy()


# ============================================================
# MODEL 2
# ============================================================

model2_X_train = model2_train[model2_features].copy()
model2_y_train = model2_train["target_ac_power"].copy()

model2_X_val = model2_val[model2_features].copy()
model2_y_val = model2_val["target_ac_power"].copy()

model2_X_test = model2_test[model2_features].copy()
model2_y_test = model2_test["target_ac_power"].copy()


# ============================================================
# VERIFICATION
# ============================================================

# print("\nMODEL 1")
# print("X_train:", model1_X_train.shape)
# print("y_train:", model1_y_train.shape)
# print("X_val  :", model1_X_val.shape)
# print("y_val  :", model1_y_val.shape)
# print("X_test :", model1_X_test.shape)
# print("y_test :", model1_y_test.shape)
#
# print("\nMODEL 2")
# print("X_train:", model2_X_train.shape)
# print("y_train:", model2_y_train.shape)
# print("X_val  :", model2_X_val.shape)
# print("y_val  :", model2_y_val.shape)
# print("X_test :", model2_X_test.shape)
# print("y_test :", model2_y_test.shape)
#
#
# print("\nMissing values:")
#
# print(
#     "Model 1:",
#     model1_X_train.isna().sum().sum()
#     + model1_X_val.isna().sum().sum()
#     + model1_X_test.isna().sum().sum()
# )
#
# print(
#     "Model 2:",
#     model2_X_train.isna().sum().sum()
#     + model2_X_val.isna().sum().sum()
#     + model2_X_test.isna().sum().sum()
# )
#
#
# print("\nFeature counts:")
# print("Model 1:", model1_X_train.shape[1])
# print("Model 2:", model2_X_train.shape[1])


# 🚀 Step 32 — Baseline Models
# print("\n" + "=" * 70)
# print("STEP 32 - PERSISTENCE BASELINE")
# print("=" * 70)

# Current-hour AC power is the prediction for next hour
baseline_pred = model2_X_test["ac_power__5069"].values

baseline_actual = model2_y_test.values

# print("\nBaseline prediction shape:", baseline_pred.shape)
# print("Actual target shape:", baseline_actual.shape)
#
# print("\nFirst 10 baseline predictions:")
# print(baseline_pred[:10])
#
# print("\nFirst 10 actual values:")
# print(baseline_actual[:10])



# Step 33:properly evaluate using:
#
# MAE
# RMSE
# R²

print("\n" + "=" * 70)
print("STEP 33 - EVALUATE PERSISTENCE BASELINE")
print("=" * 70)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

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

# print("\nPersistence Baseline Results:")
# print("MAE :", round(baseline_mae, 4))
# print("RMSE:", round(baseline_rmse, 4))
# print("R²  :", round(baseline_r2, 4))

# Step 34 — LightGBM availability check
print("\n" + "=" * 70)
print("STEP 34 - CHECK LIGHTGBM")
print("=" * 70)

try:
    import lightgbm as lgb

    print("LightGBM installed successfully!")
    print("Version:", lgb.__version__)

except ImportError:
    print("LightGBM is NOT installed.")

