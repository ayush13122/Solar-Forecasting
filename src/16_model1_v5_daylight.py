from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
import pvlib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# FIND PROJECT ROOT ROBUSTLY
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
df["target_timestamp"] = pd.to_datetime(df["target_timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)


# print("=" * 70)
# print("STEP 52 - MODEL 1 V5: DAYLIGHT SPECIALIST")
# print("=" * 70)
#
# print("\nInput shape:", df.shape)


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

# Plant DC capacity
dc_capacity_kw = 449.28


# ============================================================
# TARGET-HOUR SOLAR ZENITH
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
# DAYLIGHT / NIGHT CLASSIFICATION
# ============================================================

df["target_is_daylight"] = (
        df["target_solar_zenith"] < 90
).astype(int)

# print("\nTarget-hour regime:")
# print(
#     df["target_is_daylight"].value_counts()
#     .rename({
#         0: "Night",
#         1: "Daylight"
#     })
# )


# ============================================================
# NORMALIZED TARGET
# ============================================================

df["target_power_normalized"] = (
        df[target] / dc_capacity_kw
)


# ============================================================
# ROLLING VALIDATION FOLDS
# ============================================================

folds = [
    ("Fold 1", 2013, 2014),
    ("Fold 2", 2014, 2015),
    ("Fold 3", 2015, 2016)
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

for fold_name, train_end_year, val_year in folds:

    # print("\n" + "-" * 70)
    # print(
    #     f"{fold_name}: "
    #     f"TRAIN 2010-{train_end_year} → VALIDATE {val_year}"
    # )
    # print("-" * 70)

    train_start = pd.Timestamp("2010-01-01")

    val_start = pd.Timestamp(
        f"{val_year}-01-01"
    )

    val_end = pd.Timestamp(
        f"{val_year}-12-31 23:59:59"
    )


    # --------------------------------------------------------
    # TRAIN / VALIDATION SPLIT
    # USING TARGET TIMESTAMP
    # --------------------------------------------------------

    train_df = df[
        (df["target_timestamp"] >= train_start) &
        (df["target_timestamp"] < val_start)
        ].copy()

    val_df = df[
        (df["target_timestamp"] >= val_start) &
        (df["target_timestamp"] <= val_end)
        ].copy()


    # --------------------------------------------------------
    # DAYLIGHT TRAINING DATA ONLY
    # --------------------------------------------------------

    train_daylight = train_df[
        train_df["target_is_daylight"] == 1
        ].copy()


    X_train = train_daylight[features]

    y_train = train_daylight[
        "target_power_normalized"
    ]


    # print("\nAll train rows:", len(train_df))
    # print("Daylight train rows:", len(train_daylight))
    # print("Validation rows:", len(val_df))


    # --------------------------------------------------------
    # TRAIN DAYLIGHT MODEL
    # --------------------------------------------------------

    model = lgb.LGBMRegressor(**params)

    model.fit(
        X_train,
        y_train,
        eval_X=(
            val_df.loc[
                val_df["target_is_daylight"] == 1,
                features
            ]
        ),
        eval_y=(
            val_df.loc[
                val_df["target_is_daylight"] == 1,
                "target_power_normalized"
            ]
        ),
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(0)
        ]
    )


    # --------------------------------------------------------
    # PREDICT VALIDATION
    # --------------------------------------------------------

    val_pred_normalized = np.zeros(
        len(val_df)
    )


    daylight_mask = (
            val_df["target_is_daylight"].values == 1
    )


    if daylight_mask.sum() > 0:

        val_pred_normalized[daylight_mask] = (
            model.predict(
                val_df.loc[
                    daylight_mask,
                    features
                ],
                num_iteration=model.best_iteration_
            )
        )


    # Convert back to kW
    val_pred = (
            val_pred_normalized
            * dc_capacity_kw
    )


    actual = val_df[target].values


    # --------------------------------------------------------
    # METRICS - ALL HOURS
    # --------------------------------------------------------

    mae = mean_absolute_error(
        actual,
        val_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            val_pred
        )
    )

    r2 = r2_score(
        actual,
        val_pred
    )


    # --------------------------------------------------------
    # DAYLIGHT-ONLY METRICS
    # --------------------------------------------------------

    daylight_actual = actual[daylight_mask]
    daylight_pred = val_pred[daylight_mask]


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


    # print("\nV5 Results:")
    # print("Best iteration:", model.best_iteration_)
    # print("All-hours MAE :", round(mae, 4))
    # print("All-hours RMSE:", round(rmse, 4))
    # print("All-hours R²  :", round(r2, 4))
    # print("Daylight MAE  :", round(daylight_mae, 4))
    # print("Daylight RMSE :", round(daylight_rmse, 4))


    results.append({
        "fold": fold_name,
        "validation_year": val_year,
        "train_rows_all": len(train_df),
        "train_rows_daylight": len(train_daylight),
        "validation_rows": len(val_df),
        "all_hours_mae": mae,
        "all_hours_rmse": rmse,
        "all_hours_r2": r2,
        "daylight_mae": daylight_mae,
        "daylight_rmse": daylight_rmse
    })


# ============================================================
# SUMMARY
# ============================================================

results_df = pd.DataFrame(results)


# print("\n" + "=" * 70)
# print("MODEL 1 V5 ROLLING VALIDATION SUMMARY")
# print("=" * 70)
#
# print(
#     results_df.to_string(index=False)
# )
#
#
# print("\nAverage V5 performance:")
#
# print(
#     "All-hours MAE:",
#     round(
#         results_df["all_hours_mae"].mean(),
#         4
#     )
# )
#
# print(
#     "All-hours RMSE:",
#     round(
#         results_df["all_hours_rmse"].mean(),
#         4
#     )
# )
#
# print(
#     "All-hours R²:",
#     round(
#         results_df["all_hours_r2"].mean(),
#         4
#     )
# )
#
# print(
#     "Daylight MAE:",
#     round(
#         results_df["daylight_mae"].mean(),
#         4
#     )
# )
#
# print(
#     "Daylight RMSE:",
#     round(
#         results_df["daylight_rmse"].mean(),
#         4
#     )
# )


