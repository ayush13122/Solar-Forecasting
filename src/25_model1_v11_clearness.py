# Step 61 — V11 — Clear-Sky Normalization / Clearness Index.
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
df["target_timestamp"] = pd.to_datetime(
    df["target_timestamp"]
)

df = df.sort_values("timestamp").reset_index(drop=True)


print("=" * 70)
print("STEP 61 - MODEL 1 V11: CLEAR-SKY NORMALIZATION")
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
# SOLAR POSITION FOR CURRENT TIMESTAMP
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


# ============================================================
# CLEAR-SKY IRRADIANCE
# ============================================================

clear_sky = location.get_clearsky(
    mst_time,
    model="ineichen",
    solar_position=solar_position
)

df["clear_sky_ghi"] = clear_sky["ghi"].values


# ============================================================
# CLEARNESS INDEX
# ============================================================

# Avoid division problems around night-time / very low sun.
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


print("\nClear-sky GHI statistics:")
print(
    df["clear_sky_ghi"].describe()
)

print("\nClearness index statistics:")
print(
    df["clearness_index"].describe()
)

print(
    "\nMissing clear-sky GHI:",
    df["clear_sky_ghi"].isna().sum()
)

print(
    "Missing clearness index:",
    df["clearness_index"].isna().sum()
)


# ============================================================
# V11 FEATURES
# ============================================================

v11_features = base_features + [
    "clear_sky_ghi",
    "clearness_index"
]


print(
    "\nBase features:",
    len(base_features)
)

print(
    "V11 additional features:",
    2
)

print(
    "Total V11 features:",
    len(v11_features)
)


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
    # TARGET-TIMESTAMP BASED SPLIT
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


    X_train = train_df[v11_features]
    y_train = train_df[target]

    X_val = val_df[v11_features]
    y_val = val_df[target]


    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)

    print(
        "Missing train values:",
        X_train.isna().sum().sum()
    )

    print(
        "Missing validation values:",
        X_val.isna().sum().sum()
    )


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


    print("\nV11 Results:")

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
print("MODEL 1 V11 ROLLING VALIDATION SUMMARY")
print("=" * 70)


print(
    results_df.to_string(index=False)
)


print("\nAverage V11 performance:")

print(
    "MAE :",
    round(results_df["mae"].mean(), 4)
)

print(
    "RMSE:",
    round(results_df["rmse"].mean(), 4)
)

print(
    "R²  :",
    round(results_df["r2"].mean(), 4)
)


print("\nReference - V10:")

print("MAE :", 14.3502)
print("RMSE:", 29.9448)
print("R²  :", 0.9107)


print("\nReference - V6:")

print("MAE :", 14.7334)
print("RMSE:", 30.0998)
print("R²  :", 0.9097)