# Step 65 — V13 Multi-Weather Temporal Features
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
print("STEP 65 - MODEL 1 V13: MULTIVARIATE TEMPORAL FEATURES")
print("=" * 70)

print("\nForecast dataset:", df.shape)
print("History dataset:", history_df.shape)


# ============================================================
# CREATE TEMPORAL FEATURES FROM FULL HOURLY HISTORY
# ============================================================

hist = history_df[
    [
        "timestamp",
        "poa_irradiance__5061",
        "CLOUD_AMT",
        "ambient_temp__5062",
        "T2M",
        "RH2M",
        "WS10M",
        "ALLSKY_SFC_SW_DWN"
    ]
].copy()


# ============================================================
# POA TEMPORAL FEATURES
# ============================================================

hist["poa_lag1"] = (
    hist["poa_irradiance__5061"].shift(1)
)

hist["poa_lag3"] = (
    hist["poa_irradiance__5061"].shift(3)
)

hist["poa_change_1h"] = (
        hist["poa_irradiance__5061"]
        - hist["poa_lag1"]
)

hist["poa_change_3h"] = (
        hist["poa_irradiance__5061"]
        - hist["poa_lag3"]
)

hist["poa_rolling_mean_3h"] = (
    hist["poa_irradiance__5061"]
    .rolling(
        window=3,
        min_periods=1
    )
    .mean()
)


# ============================================================
# CLOUD TEMPORAL FEATURES
# ============================================================

hist["cloud_lag1"] = (
    hist["CLOUD_AMT"].shift(1)
)

hist["cloud_change_1h"] = (
        hist["CLOUD_AMT"]
        - hist["cloud_lag1"]
)

hist["cloud_change_3h"] = (
        hist["CLOUD_AMT"]
        - hist["CLOUD_AMT"].shift(3)
)

hist["cloud_rolling_mean_3h"] = (
    hist["CLOUD_AMT"]
    .rolling(
        window=3,
        min_periods=1
    )
    .mean()
)


# ============================================================
# TEMPERATURE TEMPORAL FEATURES
# ============================================================

hist["ambient_temp_lag1"] = (
    hist["ambient_temp__5062"].shift(1)
)

hist["ambient_temp_change_1h"] = (
        hist["ambient_temp__5062"]
        - hist["ambient_temp_lag1"]
)

hist["T2M_lag1"] = (
    hist["T2M"].shift(1)
)

hist["T2M_change_1h"] = (
        hist["T2M"]
        - hist["T2M_lag1"]
)


# ============================================================
# HUMIDITY TEMPORAL FEATURES
# ============================================================

hist["RH2M_lag1"] = (
    hist["RH2M"].shift(1)
)

hist["RH2M_change_1h"] = (
        hist["RH2M"]
        - hist["RH2M_lag1"]
)


# ============================================================
# WIND TEMPORAL FEATURES
# ============================================================

hist["WS10M_lag1"] = (
    hist["WS10M"].shift(1)
)

hist["WS10M_change_1h"] = (
        hist["WS10M"]
        - hist["WS10M_lag1"]
)


# ============================================================
# NASA IRRADIANCE TEMPORAL FEATURES
# ============================================================

hist["ALLSKY_lag1"] = (
    hist["ALLSKY_SFC_SW_DWN"].shift(1)
)

hist["ALLSKY_change_1h"] = (
        hist["ALLSKY_SFC_SW_DWN"]
        - hist["ALLSKY_lag1"]
)


# ============================================================
# TEMPORAL FEATURE LIST
# ============================================================

temporal_features = [
    "poa_lag1",
    "poa_lag3",
    "poa_change_1h",
    "poa_change_3h",
    "poa_rolling_mean_3h",

    "cloud_lag1",
    "cloud_change_1h",
    "cloud_change_3h",
    "cloud_rolling_mean_3h",

    "ambient_temp_lag1",
    "ambient_temp_change_1h",

    "T2M_lag1",
    "T2M_change_1h",

    "RH2M_lag1",
    "RH2M_change_1h",

    "WS10M_lag1",
    "WS10M_change_1h",

    "ALLSKY_lag1",
    "ALLSKY_change_1h"
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


print("\nTemporal features added:",
      len(temporal_features))

print(
    "Total rows after merge:",
    len(df)
)

print(
    "Total V13 features:",
    26 + len(temporal_features)
)


# ============================================================
# MISSING CHECK
# ============================================================

print("\nMissing temporal values:")

print(
    df[temporal_features]
    .isna()
    .sum()
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


# ============================================================
# FINAL V13 FEATURES
# ============================================================

features = (
        base_features
        + temporal_features
)

target = "target_ac_power"


print("\nBase features:", len(base_features))
print("Temporal features:", len(temporal_features))
print("Total V13 features:", len(features))


# ============================================================
# RECENT-HISTORY ROLLING FOLDS
# ============================================================

folds = [
    ("Fold 1", 2012, 2013, 2014),
    ("Fold 2", 2013, 2014, 2015),
    ("Fold 3", 2014, 2015, 2016)
]


# ============================================================
# LIGHTGBM PARAMETERS
# ============================================================

params = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": 42,
    "verbosity": -1
}


results = []


# ============================================================
# ROLLING VALIDATION
# ============================================================

for (
        fold_name,
        train_start_year,
        train_end_year,
        val_year
) in folds:

    print("\n" + "-" * 70)

    print(
        f"{fold_name}: "
        f"TRAIN {train_start_year}-{train_end_year} "
        f"→ VALIDATE {val_year}"
    )

    print("-" * 70)


    train_start = pd.Timestamp(
        f"{train_start_year}-01-01"
    )

    train_end = pd.Timestamp(
        f"{train_end_year}-12-31 23:59:59"
    )

    val_start = pd.Timestamp(
        f"{val_year}-01-01"
    )

    val_end = pd.Timestamp(
        f"{val_year}-12-31 23:59:59"
    )


    # --------------------------------------------------------
    # TARGET-TIMESTAMP SPLIT
    # --------------------------------------------------------

    train_df = df[
        (df["target_timestamp"] >= train_start)
        &
        (df["target_timestamp"] <= train_end)
        ].copy()

    val_df = df[
        (df["target_timestamp"] >= val_start)
        &
        (df["target_timestamp"] <= val_end)
        ].copy()


    X_train = train_df[features]
    y_train = train_df[target]

    X_val = val_df[features]
    y_val = val_df[target]


    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model = lgb.LGBMRegressor(
        **params
    )


    model.fit(
        X_train,
        y_train,
        eval_X=X_val,
        eval_y=y_val,
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(0)
        ]
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    val_pred = model.predict(
        X_val,
        num_iteration=model.best_iteration_
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_val,
        val_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_val,
            val_pred
        )
    )

    r2 = r2_score(
        y_val,
        val_pred
    )


    print("\nV13 Results:")

    print(
        "Best iteration:",
        model.best_iteration_
    )

    print(
        "MAE :",
        round(mae, 4)
    )

    print(
        "RMSE:",
        round(rmse, 4)
    )

    print(
        "R²  :",
        round(r2, 4)
    )


    results.append({
        "fold": fold_name,
        "validation_year": val_year,
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "best_iteration": model.best_iteration_
    })


# ============================================================
# SUMMARY
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n" + "=" * 70)
print("MODEL 1 V13 ROLLING VALIDATION SUMMARY")
print("=" * 70)


print(
    results_df.to_string(index=False)
)


print("\nAverage V13 performance:")

print(
    "MAE :",
    round(
        results_df["mae"].mean(),
        4
    )
)

print(
    "RMSE:",
    round(
        results_df["rmse"].mean(),
        4
    )
)

print(
    "R²  :",
    round(
        results_df["r2"].mean(),
        4
    )
)


print("\nReference - V6:")

print("MAE :", 14.7334)
print("RMSE:", 30.0998)
print("R²  :", 0.9097)


print("\nReference - V10:")

print("MAE :", 14.3502)
print("RMSE:", 29.9448)
print("R²  :", 0.9107)