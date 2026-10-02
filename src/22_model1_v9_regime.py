# Step 58 — V9 Regime-Specific LightGBM
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
print("STEP 58 - MODEL 1 V9: REGIME-SPECIFIC MODEL")
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


# ============================================================
# REGIME DEFINITION
# ============================================================

# Regime 0 = low / near-zero irradiance
# Regime 1 = active generation conditions

df["regime"] = (
        df["poa_irradiance__5061"] > 100
).astype(int)


print("\nRegime distribution:")
print(
    df["regime"]
    .value_counts()
    .sort_index()
    .rename({
        0: "Low irradiance (<=100)",
        1: "Active generation (>100)"
    })
)


# ============================================================
# ROLLING VALIDATION
# ============================================================

folds = [
    ("Fold 1", 2010, 2013, 2014),
    ("Fold 2", 2010, 2014, 2015),
    ("Fold 3", 2010, 2015, 2016)
]


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
# FOLD LOOP
# ============================================================

for fold_name, train_start_year, train_end_year, val_year in folds:

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
    # SPLIT BY TARGET TIMESTAMP
    # --------------------------------------------------------

    train_df = df[
        (df["target_timestamp"] >= train_start) &
        (df["target_timestamp"] <= train_end)
        ].copy()

    val_df = df[
        (df["target_timestamp"] >= val_start) &
        (df["target_timestamp"] <= val_end)
        ].copy()


    print("Train rows:", len(train_df))
    print("Validation rows:", len(val_df))


    # --------------------------------------------------------
    # CREATE TWO SPECIALIST MODELS
    # --------------------------------------------------------

    model_low = lgb.LGBMRegressor(**params)

    model_active = lgb.LGBMRegressor(**params)


    # ========================================================
    # LOW-IRRADIANCE MODEL
    # ========================================================

    train_low = train_df[
        train_df["regime"] == 0
        ]

    val_low = val_df[
        val_df["regime"] == 0
        ]


    if len(train_low) > 0 and len(val_low) > 0:

        model_low.fit(
            train_low[features],
            train_low[target],
            eval_X=val_low[features],
            eval_y=val_low[target],
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(0)
            ]
        )


    # ========================================================
    # ACTIVE-GENERATION MODEL
    # ========================================================

    train_active = train_df[
        train_df["regime"] == 1
        ]

    val_active = val_df[
        val_df["regime"] == 1
        ]


    if len(train_active) > 0 and len(val_active) > 0:

        model_active.fit(
            train_active[features],
            train_active[target],
            eval_X=val_active[features],
            eval_y=val_active[target],
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(0)
            ]
        )


    # ========================================================
    # ROUTE VALIDATION ROWS TO SPECIALIST MODELS
    # ========================================================

    val_pred = np.zeros(len(val_df))

    low_mask = (
            val_df["regime"].values == 0
    )

    active_mask = (
            val_df["regime"].values == 1
    )


    if low_mask.sum() > 0:
        val_pred[low_mask] = model_low.predict(
            val_df.loc[low_mask, features],
            num_iteration=model_low.best_iteration_
        )


    if active_mask.sum() > 0:
        val_pred[active_mask] = model_active.predict(
            val_df.loc[active_mask, features],
            num_iteration=model_active.best_iteration_
        )


    # ========================================================
    # ALL-HOURS METRICS
    # ========================================================

    actual = val_df[target].values

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


    # ========================================================
    # ACTIVE-GENERATION METRICS
    # ========================================================

    active_actual = actual[active_mask]
    active_pred = val_pred[active_mask]


    active_mae = mean_absolute_error(
        active_actual,
        active_pred
    )

    active_rmse = np.sqrt(
        mean_squared_error(
            active_actual,
            active_pred
        )
    )


    print("\nV9 Results:")
    print("Low-regime train rows:", len(train_low))
    print("Active-regime train rows:", len(train_active))

    print(
        "All-hours MAE :",
        round(mae, 4)
    )

    print(
        "All-hours RMSE:",
        round(rmse, 4)
    )

    print(
        "All-hours R²  :",
        round(r2, 4)
    )

    print(
        "Active-regime MAE:",
        round(active_mae, 4)
    )

    print(
        "Active-regime RMSE:",
        round(active_rmse, 4)
    )


    results.append({
        "fold": fold_name,
        "validation_year": val_year,
        "validation_rows": len(val_df),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "active_mae": active_mae,
        "active_rmse": active_rmse
    })


# ============================================================
# SUMMARY
# ============================================================

results_df = pd.DataFrame(results)


print("\n" + "=" * 70)
print("MODEL 1 V9 ROLLING VALIDATION SUMMARY")
print("=" * 70)

print(
    results_df.to_string(index=False)
)


print("\nAverage V9 performance:")

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

print(
    "Active MAE:",
    round(results_df["active_mae"].mean(), 4)
)

print(
    "Active RMSE:",
    round(results_df["active_rmse"].mean(), 4)
)


print("\nReference - V6:")

print("MAE :", 14.7334)
print("RMSE:", 30.0998)
print("R²  :", 0.9097)